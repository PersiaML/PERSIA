r"""

.. class:: PersiaBatchDataChannel(buffer_size: int)

    This class is a rust_extension pyclass that create the :class:`.PersiaBatchDataReceiver` 
    and the :class:`.PersiaBatchDataSender`. It can transfer the :class:`.PersiaBatch` 
    from the Python to Rust.

    :param int buffer_size: buffer size of the :class:`.PersiaBatchDataReceiver` and the 
        :class:`.PersiaBatchDataSender`.

    .. method:: get_receiver(self) 

        Get the ``PersiaBatchDataReceiver``

        :rtype: PersiaBatchDataReceiver
    
    .. method:: get_sender(self)

        Get the ``PersiaBatchDataSender``

        :rtype: PersiaBatchDataSender

.. class:: PersiaBatchDataSender

    This ``class`` cannot be instantiate directly. It can send the ``PersiaBatch``  to 
    the ``PersiaBatchDataReceiver``.

    .. method:: send(persia_batch: PersiaBatch)

        Send the ``PersiaBatch`` data to ``PersiaBatchDataReceiver``.

.. class:: PersiaBatchDataReceiver
    
    This ``class`` cannot be instantiate directly. It can receive the ``PersiaBatch`` 
    from the ``PersiaBatchDataSender``.

"""
from abc import ABC, abstractmethod
from collections.abc import Iterable as IterableABC
from threading import Thread
from typing import Iterator, Optional

import persia.env as env

from persia.ctx import cnt_ctx
from persia.embedding.data import PersiaBatch
from persia.logger import get_default_logger
from persia.prelude import (
    PersiaBatchDataChannel,
    initialize_dataflow,
    Forward,
)

_logger = get_default_logger()


class IterableDatasetBase(ABC, IterableABC[PersiaBatch]):
    r"""IterableDataSet base wrap the :class:`PersiaBatchDataChannel` that provide the sender
    and receiver of the channel.

    The role of this `class` is to transfer the :class:`.PersiaBatch` to the
    :class:`.DataLoader`. This class cannot be used directly unless it implements
    the ``__iter__`` and ``consume_dataset`` function to ensure the
    :class:`.DataLoader` works fine.

    Implements the ``__iter__`` function to generate the :class:`.PersiaBatch`. And
    implements the ``consume_dataset`` to send the :class:`.PersiaBatch` by
    :class:`.PersiaBatchDataSender`.

    Here is an example that implements the synchronously ``IterableDatasetBase``.

    .. code-block:: python

        from typing import Iterator

        import numpy as np
        from persia.data import IterableDataset
        from persia.embedding.data import PersiaBatch, IDTypeFeature

        class MyPersiaIterableDataset(IterableDatasetBase):

            def __iter__(self):
                persia_batch = PersiaBatch(id_type_features=IDTypeFeature(
                    "id_type_feature_slot",
                    [
                        np.array([1000, 10001], dtype=np.uint64),
                        np.array([1003, 10011], dtype=np.uint64),
                    ]
                ), requires_grad=False)

                yield persia_batch
                yield persia_batch

            def consume_data(self) -> Iterator[int]:
                for preprocess_idx, persia_batch in enumerate(self):
                    self.sender.send(persia_batch)
                    yield preprocess_idx

    .. note::
        The above example can not meet the performance requirement if you face a large dataset
        due to it processing the :class:`.PersiaBatch` synchronously. If you want to improve
        the performance of data processing, try to use the :class:`.IterableDataset` or
        :class:`.StreamingDataset` instead.

    Arguments:
        buffer_size (int, optional): buffer size for :class:`PersiaBatchDataChannel`.
    """

    def __init__(
        self,
        buffer_size: int = 10,
    ):
        self.persia_batch_channel = PersiaBatchDataChannel(buffer_size)
        self.sender = self.persia_batch_channel.get_sender()
        self.receiver = self.persia_batch_channel.get_receiver()

    @abstractmethod
    def consume_dataset(self) -> Iterator[int]:
        """Consume ``__iter__`` of itself and return the iterator of preprocess indexes."""
        ...


class StreamingDataset(IterableDatasetBase):
    r"""Streaming dataset receives the :class:`PersiaBatch` from the upstream data
    flow that sent by :class:`.DataCtx`.

    In the implemented :meth:`.StreamingDataset.consume_dataset`, the
    :class:`PersiaBatchDataSender` instance is bind into the RPC service that receive the
    data automatically. So it is not necessary to implements the

    .. warning::
        :class:`.StreamingDataset` will make the :class:`.DataLoader` raise the
        ``TimeoutError`` if the upstream data flow drained.

    Arguments:
        buffer_size (int, optional): :class:`PersiaBatchDataChannel` buffer size
    """

    def __init__(
        self,
        buffer_size: int = 10,
    ):
        super(StreamingDataset, self).__init__(buffer_size)
        self.initialized = False

    def consume_dataset(self) -> Iterator[int]:
        if not self.initialized:
            world_size = env.get_world_size()
            assert world_size, "WORLD_SIZE cannot be None"
            initialize_dataflow(world_size, self.sender)

            _logger.info("initialize the streaming dataset.")
            self.initialized = True

        idx = 0
        while True:
            yield idx
            idx += 1

    def __iter__(self):
        raise NotImplementedError(
            "StreamingDataset can not be iterable directly, you can use it\
            combine with persia.data.DataLoader."
        )


class IterableDataset(IterableDatasetBase):
    r"""``IterableDataset`` can iterate the dataset multiple times compare to
    :class:`.StreamingDataset`. so you can implement the TestDataset based on
    ``IterableDataset``.

    Implements the ``__iter__`` function to define the :class:`PersiaBatch` generation phase.

    .. code-block:: python

        import numpy as np
        from persia.data import IterableDataset, DataLoader
        from persia.embedding.data import PersiaBatch, IDTypeFeature

        class TestDataset(IterableDataset):
            def __init__(self):
                super(MyTestDataset, self).__init__()
                self.data = data
                self.size = 10

            def __iter__(self):
                for i in range(self.size):
                    persia_batch = PersiaBatch(id_type_features=IDTypeFeature(
                        "id_type_feature_slot",
                        [
                            np.array([1000, 10001], dtype=np.uint64),
                            np.array([1003, 10011], dtype=np.uint64),
                        ]
                    ), requires_grad=False)
                    yield persia_batch

        dataset = MyTestDataset()
        dataloader = DataLoader(dataset)

    Arguments:
        buffer_size (int, optional): :class:`.PersiaBatch` buffer size
    """

    def consume_dataset(self) -> Iterator[int]:
        from queue import Queue

        preprocess_queue = Queue()

        def send_data():
            for preprocess_idx, persia_batch in enumerate(self):
                self.sender.send(persia_batch)
                preprocess_queue.put(preprocess_idx)
            preprocess_queue.put(None)  # end the iteration

        handler = Thread(target=send_data, daemon=True)
        handler.start()

        while True:
            preprocess_idx = preprocess_queue.get()
            if preprocess_idx is not None:
                yield preprocess_idx
            else:
                break

        handler.join()


class DataLoader:
    r"""Data loader will preprocess the data to the ``PersiaTrainingBatch``.

    The :class:`DataLoader` is a pipeline that preprocess the :class:`.PersiaBatch` in
    several steps. Each step will process the task concurrently with multiple threads
    to improve the efficiency.

    .. warning::
        The :class:`DataLoader` cannot stop the iteration unless raise the ``TimeoutError``
        if you use the :class:`.StreamingDataset`.

    Arguments:
        dataset (IterableDatasetBase): dataset for DataLoader to retrive replica info
            and sender channel.
        forward_buffer_size: (int, optional): ``PersiaTrainingBatch`` buffer size, this
            args effect the gpu memory cost.
        is_training (bool, optional): whether current forward status is training or not.
        timeout_ms (int, optional): timeout for Forward to fetch data, millisecond unit.
        num_workers (int, optional): spawn thread worker number for Forward to lookup
            embedding and :class:`.PersiaBatch` prefetch.
        reproducible (bool, optional): iterate the data in fixed order, make the dataflow
            deterministic.
        embedding_staleness (int, optional): max number of batched staleness embedding each
            rank. A staleness embedding means it prefetched from embedding server before
            gradient updated.
    """

    def __init__(
        self,
        dataset: IterableDatasetBase,
        forward_buffer_size: int = 10,
        is_training: bool = True,
        timeout_ms: int = 1000 * 60 * 10,
        num_workers: int = 10,
        reproducible: bool = False,
        embedding_staleness: Optional[int] = None,
    ):
        assert isinstance(
            dataset, IterableDatasetBase
        ), "dataset invalid, you should use persia.data.IterableDatasetBase class\
            to generate the data."

        self.dataset = dataset
        self.timeout_ms = timeout_ms
        self.num_workers = num_workers

        self.forward_engine = Forward(
            forward_buffer_size,
            is_training,
            reproducible,
            embedding_staleness,
        )

        self.launch = False
        self.forward_engine.set_input_channel(dataset.receiver)

    def __iter__(self):

        if not self.launch:
            current_ctx = cnt_ctx()
            assert current_ctx is not None, "Current conext is None!"

            self.forward_engine.launch(self.num_workers)
            self.launch = True

        try:
            for preprocess_idx in self.dataset._consume_dataset():
                yield self.forward_engine.get_batch(self.timeout_ms)
        except TimeoutError:
            _logger.warning(
                f"get_batch time out, stop iter data, current batch_idx: {preprocess_idx}"
            )

    def __del__(self):
        self.forward_engine.shutdown()
