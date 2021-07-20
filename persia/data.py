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


class IterableChannelBase(ABC):
    r"""IterableChannelBase wrap the PyPersiaBatchDataChannel that provide the channel sender and
    receiver.

    Arguments:
        buffer_size (int): PyPersiaBatchDataChannel buffer size
        replica_info (PyPersiaReplicaInfo): replica info of current process to enable the data reorder ability
    """

    def __init__(self, buffer_size: int, replica_info: PyPersiaReplicaInfo):
        self.persia_batch_channel = PyPersiaBatchDataChannel(buffer_size)
        self.replica_info = replica_info

    @property
    def receiver(self) -> PyPersiaBatchDataReceiver:
        """Get PyPersiaBatchDataReceiver python wrapper"""
        return self.persia_batch_channel.get_receiver()

    @property
    def sender(self) -> PyPersiaBatchDataSender:
        """Get PyPersiaBatchDataSender python wrapper"""
        return self.persia_batch_channel.get_sender()

    @abstractmethod
    def __iter__(self):
        """Implement this function to iterate data from PyForward"""
        ...

    @abstractmethod
    def __len__(self):
        """Fixed size dataset should implement this function"""
        ...


class NatsStreamingChannel(IterableChannelBase):
    r"""NatsStreamingChannel  recive data from nats publisher

    Arguments:
        buffer_size (int): PyPersiaBatchDataChannel buffer size
        replica_info (PyPersiaReplicaInfo): replica info of current process to enable the data reorder ability
    """

    def __init__(
        self,
        buffer_size: int,
        replica_info: PyPersiaReplicaInfo,
    ):
        super(NatsStreamingChannel, self).__init__(buffer_size, replica_info)
        self._responder = PyPersiaBatchFlowNatsStubResponder(replica_info, self.sender)

    def __iter__(self):
        while True:
            yield None

    def __len__(self) -> int:
        raise NotImplementedError("StreamingChannel do not implement __len__ function")


class PersiaChannel(IterableChannelBase):
    r"""Persia data channel that provide asynchronous data handler feature to improve the performance of data preprocess.
    Not support synchronous data handler temporary.

    Arguments:
        buffer_size (int): PyPersiaBatchDataChannel buffer size
        replica_info (PyPersiaReplicaInfo): replica info of current process to enable the data reorder ability
    """

    def __init__(
        self,
        buffer_size: int,
        replica_info: PyPersiaReplicaInfo = None,
        async_iterator: bool = True,
    ):
        super(PersiaChannel, self).__init__(buffer_size, replica_info)
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

        # TODO: process the forward engine error to prevent blocking
        for _val in range(len(self)):
            yield _val

        if self.async_iterator:
            handler.join()


class Dataloder(object):
    r"""Dataloder provide the interface to fetch the PythonBatchData from PyForward
    wrapper.

    Arguments:
        channel (IterableChannelBase): data channel for Dataloder to retrive replica info and sender channel
        forward_buffer_size: (int, optional): gpu forward channel buffer size, this args effect the gpu memory cost
        is_training (bool, optional): wheter current forward status is training or not
        timeout (int, optional): timeout for PyFoward to fetch data, millisecond unit
        num_workers (int, optional): spawn thread worker number for  PyForward to lookup embedding and PythonBatchData prefetch
        reproducible (bool, optional): iterate the data in order, make the dataflow deterministic
    """

    def __init__(
        self,
        channel: IterableChannelBase,
        forward_buffer_size: int = 10,
        is_training: bool = True,
        timeout: int = 1000 * 60 * 10,
        num_workers: int = 10,
        reproducible: bool = False,
    ):
        # dynamic import the PyForward due to conditional compilation
        from persia.prelude import PyForward

        self.channel = channel
        self.timeout = timeout
        self.num_workers = num_workers

        self.forward_engine = PyForward(
            forward_buffer_size,
            is_training,
            reproducible,
            channel.replica_info,
        )
        self.forward_engine.set_input_channel(channel.receiver)
        self.forward_engine.launch(
            torch.cuda.current_device(),
            self.num_workers,
        )

    def __iter__(self):

        for _ in self.channel:
            yield self.forward_engine.get_batch(self.timeout)

    def __len__(self):
        return len(self.channel)
