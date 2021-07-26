from threading import Thread

import torch
from torch.utils.data.dataset import IterableDataset as TorchIterableDataset

from persia.ctx import cnt_ctx
from persia.logger import get_default_logger
from persia.prelude import (
    PyPersiaBatchDataChannel,
    PyPersiaReplicaInfo,
    PyPersiaBatchDataReceiver,
    PyPersiaBatchDataSender,
    init_responder,
)

_logger = get_default_logger()


class IterableDataset(
    TorchIterableDataset
):  # TODO: cannot understand the whole doc string
    r"""IterableChannelBase wrap the PyPersiaBatchDataChannel that provide the channel sender and
    receiver.

    Arguments:
        buffer_size (int): PyPersiaBatchDataChannel buffer size
    """

    def __init__(self, buffer_size: int):
        self.persia_batch_channel = PyPersiaBatchDataChannel(buffer_size)

    @property
    def receiver(self) -> PyPersiaBatchDataReceiver:
        """Get PyPersiaBatchDataReceiver python wrapper"""
        return self.persia_batch_channel.get_receiver()

    @property
    def sender(self) -> PyPersiaBatchDataSender:
        """Get PyPersiaBatchDataSender python wrapper"""
        return self.persia_batch_channel.get_sender()


class StreamingDataset(IterableDataset):
    r"""NatsStreamingChannel receive data from nats publisher

    Arguments:
        buffer_size (int): PyPersiaBatchDataChannel buffer size
    """

    def __init__(
        self,
        buffer_size: int,
    ):
        super(StreamingDataset, self).__init__(buffer_size)
        self.initialized = False

    def __iter__(self):
        if not self.initialized:
            current_ctx = cnt_ctx()
            assert current_ctx is not None, "Current conext is None!"
            replica_info = current_ctx.replica_info
            init_responder(replica_info, self.sender)

            _logger.info("initialize the responder")
            self.initialized = True

        while True:
            yield None


class PersiaDataset(IterableDataset):
    r"""Persia data channel that provide asynchronous data handler feature to improve the performance of data preprocess.
    Not support synchronous data handler temporary.

    Arguments:
        buffer_size (int): PyPersiaBatchDataChannel buffer size
        async_iterator (bool, optional): launch the thread to generate the data asynchronous
    """

    def __init__(
        self,
        buffer_size: int,
        async_iterator: bool = True,
    ):
        super(PersiaDataset, self).__init__(
            buffer_size,
        )
        self.async_iterator = async_iterator

    def fetch_data(self, sender: PyPersiaBatchDataSender):
        """Callback function to put the data into PyPersiaBatchDataSender

        Arguments:
            sender (PyPersiaBatchDataSender): PersiaBatchData sender channel to send the generate data
                to the PersiaBatchData receive channel
        """
        raise NotImplementedError("implement this function to fetch data")

    def __iter__(self):
        # TODO: provide process worker to handler the data with high performance calculating
        # class WorkerType(Enum):
        #     THREAD = 1
        #     PROCESS = 2

        if self.async_iterator:
            handler = Thread(target=self.fetch_data, args=(self.sender,), daemon=True)
            handler.start()

        for _val in range(len(self)):
            yield _val

        if self.async_iterator:
            handler.join()


class Dataloder(object):
    r"""Dataloder provide the interface to fetch the PythonBatchData from PyForward
    wrapper.

    Arguments:
        dataset (IterableChannelBase): dataset for Dataloder to retrive replica info and sender channel
        forward_buffer_size: (int, optional): gpu forward channel buffer size, this args effect the gpu memory cost
        is_training (bool, optional): whether current forward status is training or not
        timeout (int, optional): timeout for PyForward to fetch data, millisecond unit
        num_workers (int, optional): spawn thread worker number for  PyForward to lookup embedding and PythonBatchData prefetch
        reproducible (bool, optional): iterate the data in fixed order, make the dataflow deterministic
    """

    def __init__(
        self,
        dataset: IterableDataset,
        forward_buffer_size: int = 10,
        is_training: bool = True,
        timeout: int = 1000 * 60 * 10,
        num_workers: int = 10,
        replica_info: PyPersiaReplicaInfo = None,
        reproducible: bool = False,
    ):
        # dynamic import the PyForward due to conditional compilation
        from persia.prelude import PyForward

        self.dataset = dataset
        self.timeout = timeout
        self.num_workers = num_workers

        if not replica_info:
            current_ctx = cnt_ctx()
            assert current_ctx is not None, "Current conext is None!"
            replica_info = current_ctx.replica_info

        self.forward_engine = PyForward(
            forward_buffer_size, is_training, reproducible, replica_info
        )
        self.forward_engine.set_input_channel(dataset.receiver)
        self.forward_engine.launch(
            torch.cuda.current_device(),
            self.num_workers,
        )

    def __iter__(self):

        for _ in self.dataset:
            yield self.forward_engine.get_batch(self.timeout)

    def __len__(self):
        return len(self.dataset)
