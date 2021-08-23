from typing import List

from persia.prelude import (
    PyPersiaRpcClient,
    PyPersiaBatchData,
    PyPersiaReplicaInfo,
    PyPersiaBatchFlowNatsStubPublisher,
    PyOptimizerBase,
)
from persia.error import PersiaRuntimeException
from persia.env import get_world_size

_backend = None


class Backend:
    r"""Backend class is the python wrapper of PersiaRpcClient and NatsPublisher that provide the ability to
    invoke rpc function to communicate with the trainer and middleware componenet.

    Arguments:
        threadpool_worker_size (int): Rpc client threadpool size.
        replica_info (PyPersiaReplicaInfo): Replica info of current process.
    """

    def __init__(self, threadpool_worker_size: int, replica_info: PyPersiaReplicaInfo):
        self.rpc_client = PyPersiaRpcClient(threadpool_worker_size)

        world_size = get_world_size()
        self.nats_publisher = PyPersiaBatchFlowNatsStubPublisher(
            replica_info, world_size if world_size != -1 else None
        )
        self.nats_publisher.wait_servers_ready()

    def send_data(self, data: PyPersiaBatchData, blocking: bool = True):
        """Send PersiaBatchData from data compose to trainer and middleware side.

        Arguments:
            data (PyPersiaBatchData): PersiaBatchData that haven't been process.
            blocking (bool, optional): Wait util the data send successfully.
        """
        self.nats_publisher.send_sparse_to_middleware(data, blocking)
        self.nats_publisher.send_dense_to_trainer(data, blocking)

    def get_embedding_size(self) -> List[int]:
        """Get current embedding server embedding size"""

        return self.rpc_client.get_embedding_size()

    def set_configuration(
        self,
        initialize_lower: float,
        initialize_upper: float,
        admit_probability: float,
        enable_weight_bound: bool,
        weight_bound: float,
    ):
        """Set the configuration of embedding initialization and embedding update rule to embedding server.

        Arguments:
            initialize_lower (float): embedding uniform initialization lower bound args.
            initialize_upper (float): embedding uniform initialization upper bound args.
            admit_probability (float): probability of embedding generation. value in [0, 1]. Generate a random value(range in [0, 1]) for each sparse embedding,
                    generate the new embedding once the value is small than the admit_probability. Always generate the new embedding when the admit_probability set
                    to 1.
            enable_weight_bound (bool): Enable weight bound or not.
            weight_bound (float): Restrict each element value of an embedding in [-weight_bound, weight_bound].
        """
        self.nats_publisher.configure_sharded_servers(
            initialize_lower,
            initialize_upper,
            admit_probability,
            enable_weight_bound,
            weight_bound,
        )

    def register_optimizer(self, optimizer: PyOptimizerBase):
        """Register the sparse optimizer to embedding server.

        Arguments:
            optimizer (PyOptimizerBase): optimizer python wrapper.
        """
        self.nats_publisher.register_optimizer(optimizer)

    def dump_embedding(self, dst_dir: str, blocking: bool = False):
        """Dump the sparse embedding to destination directory.This method provide not blocking
        mode, invoke the TrainCtx.wait_for_dump_embedding once set blocking to False.

        Arguments:
            dst_dir (str): Destination directory.
            blocking (bool, optional): Dump embedding util finished if `blocking` set to `True`.
        """
        self.rpc_client.dump_embedding(dst_dir)
        if blocking:
            self.rpc_client.wait_for_dump_embedding()

    def load_embedding(self, src_dir: str, blocking: bool = True):
        """Load the sparse embedding from source directory by invoke rpc client.This method provide
        not blocking mode,invoke the TrainCtx.wait_for_load_embedding once set the blocking to False.

        Arguments:
            src_dir (str): Source directory.
            blocking (bool, optional): Load embedding util finished if `blocking` set to `True`.
        """
        self.rpc_client.load_embedding(src_dir)
        if blocking:
            self.rpc_client.wait_for_load_embedding()

    def wait_for_dump_embedding(self):
        """Wait for dump the sparse embedding."""
        self.rpc_client.wait_for_dump_embedding()

    def wait_for_load_embedding(self):
        """Wait for load the sparse embedding."""
        self.rpc_client.wait_for_load_embedding()


def init_backend(
    threadpool_worker_size: int,
    replica_info: PyPersiaReplicaInfo,
) -> Backend:
    """Initialized the singleton Backend instance.

    Arguments:
        threadpool_worker_size (int, optional): Rpc client threadpool size.
        replica_info (PyPersiaReplicaInfo, optional): Replica info of current process.
    """
    global _backend
    if not _backend:
        _backend = Backend(threadpool_worker_size, replica_info)
    return _backend


def get_backend() -> Backend:
    """Get singleton Backend instance, raise PersiaRuntimeException once uninitialized the backend."""
    if not _backend:
        raise PersiaRuntimeException("init persia backend first")
    return _backend
