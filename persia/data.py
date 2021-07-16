from abc import ABC, abstractmethod
from threading import Thread

import torch


from persia.prelude import (
    PyPersiaBatchDataChannel,
    PyPersiaReplicaInfo,
    PyPersiaBatchDataReceiver,
    PyPersiaBatchDataSender,
    PyPersiaBatchFlowNatsStubResponder,
)


class IterableChanneltBase(ABC):
    def __init__(self, buffer_size: int, replica_info: PyPersiaReplicaInfo):
        self.persia_batch_channel = PyPersiaBatchDataChannel(buffer_size)
        self.replica_info = replica_info

    @property
    def receiver(self) -> PyPersiaBatchDataReceiver:
        return self.persia_batch_channel.get_receiver()

    @property
    def sender(self) -> PyPersiaBatchDataReceiver:
        return self.persia_batch_channel.get_sender()

    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __len__(self):
        ...


class NatsInfiniteDataset(IterableChanneltBase):
    r"""InfiniteIterator for streaming data stop by timeout exception

    Arguments:
        forward_engine : rust forward engine wrapper for fetch input data
        port (int): port for input server to bind
        data_queue_size (int): buffer size for data forward phase
        timeout (int): timeout for data fetch
    """

    def __init__(
        self,
        buffer_size: int,
        replica_info: PyPersiaReplicaInfo,
    ):
        super(NatsInfiniteDataset, self).__init__(buffer_size, replica_info)
        self._responder = PyPersiaBatchFlowNatsStubResponder(replica_info, self.sender)

    def __iter__(self):
        while True:
            yield None

    def __len__(self) -> int:
        raise NotImplementedError("InfiteIterator not implement __len__ function")


class FiniteDataset(IterableChanneltBase):
    def __init__(
        self,
        buffer_size: int,
        replica_info: PyPersiaReplicaInfo = None,
        async_iterator: bool = True,
    ):
        super(FiniteDataset, self).__init__(buffer_size, replica_info)
        self.async_iterator = async_iterator

    def fetch_data(self, sender: PyPersiaBatchDataSender):
        raise NotImplementedError("implement this function to fetch data")

    def __iter__(self):
        # TODO: provide process worker to handler the data with high performance calculating
        # class WorkerType(Enum):
        #     THREAD = 1
        #     PROCESS = 2

        if self.async_iterator:
            handler = Thread(target=self.fetch_data, args=(self.sender,), daemon=True)
            handler.start()
            
        # TODO: process the forward engine error to prevent blocking
        for _val in range(len(self)):
            yield _val

        if self.async_iterator:
            handler.join()


class Dataloder(object):
    def __init__(
        self,
        channel: IterableChanneltBase,
        forward_buffer_size: int = 10,
        is_training: bool = True,
        timeout: int = 1000 * 60 * 10,
        num_workers: int = 10,
        shuffle: bool = False,
    ):
        # dynamic import the PyForward due to conditional compilation
        from persia.prelude import PyForward

        self.channel = channel
        self.timeout = timeout
        self.num_workers = num_workers

        self.forward_engine = PyForward(
            forward_buffer_size,
            not is_training,
            not shuffle,
            channel.replica_info,
        )
        self.forward_engine.set_input_channel(channel.receiver)
        self.forward_engine.launch(
            torch.cuda.current_device(),
            self.num_workers,
        )

    def __iter__(self):

        # TODO: warp the rust exception to python Exception
        for _ in self.channel:
            yield self.forward_engine.get_batch(self.timeout)

    def __len__(self):
        return len(self.channel)
