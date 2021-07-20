from persia.prelude import (
    PyPersiaRpcClient,
    PyPersiaBatchData,
    PyPersiaReplicaInfo,
    PyPersiaBatchFlowNatsStubPublisher,
    PyOptimizerBase,
)
from persia.error import PersiaRuntimeException

_backend = None


class Backend:
    r"""Backend class is the PersiaRpcClient and NatsPublisher python wrapper that provide the ability to
    invoke rpc function

    Arguments:
        worker_size (int): rpc client thread pool size
        replica_info (PyPersiaReplicaInfo): replica info of current process.
    """

    def __init__(
        self,
        worker_size: int,
        replica_info: PyPersiaReplicaInfo,
    ):
        self.rpc_client = PyPersiaRpcClient(worker_size)
        self.nats_publisher = PyPersiaBatchFlowNatsStubPublisher(replica_info)
        self.nats_publisher.wait_servers_ready()

    def send_data(self, data: PyPersiaBatchData, blocking: bool = True):
        """Send data from data compose to trainer and middleware side

        Arguments:
            data (PyPersiaBatchData): persia_batch_data
            blocking (bool): whether retry sending data when meet exception.RuntimeExcption will raised directly
                without retry when set it to False.
        """
        self.nats_publisher.send_sparse_to_middleware(data, blocking)
        self.nats_publisher.send_dense_to_trainer(data, blocking)

    def set_configuration(
        self,
        initialize_lower: float,
        initialize_upper: float,
        admit_probability: float,
        enable_weight_bound: bool,
        weight_bound: float,
    ):
        """Set embedding server configuration

        Arguments:
            initialize_lower (float): embedding uniform initialization lower bound args
            initialize_upper (float): embedding uniform initialization upper bound args
            admit_probability (float): probability of embedding generation. value in [0, 1]. Generate a random value(range in [0, 1]) for each sparse embedding,
                    generate the new embedding once the value is small than the admit_probability. Always generate the new embedding when the admit_probability set
                    to 1.
        """
        self.nats_publisher.configure_sharded_servers(
            initialize_lower,
            initialize_upper,
            admit_probability,
            enable_weight_bound,
            weight_bound,
        )

    def register_optimizer(self, optimizer: PyOptimizerBase):
        """Register the Optimizer by nats publisher

        Arguments:
            optimizer (PyOptimizerBase): optimizer python wrapper
        """
        self.nats_publisher.register_optimizer(optimizer)

    def dump_embedding(self, dst_dir: str, blocking: bool = False):
        """Dump the sparse embedding to destination directory.This method provide not blocking
        mode, invoke the TrainCtx.wait_for_dump_embedding once set blocking to False.

        Arguments:
            dst_dir (str): destination directory
            blocking (bool, optional): dump embedding in blocking mode or not
        """
        self.rpc_client.dump_embedding(dst_dir)
        if blocking:
            self.rpc_client.wait_for_dump_embedding()

    def load_embedding(self, src_dir: str, blocking: bool = True):
        """Load the sparse embedding from source directory by invoke rpc client.This method provide
        not blocking mode,invoke the TrainCtx.wait_for_load_embedding once set the blocking to False.

        Arguments:
            src_dir (str): destination directory
            blocking (bool, optional): dump embedding in blocking mode or not
        """
        self.rpc_client.load_embedding(src_dir)
        if blocking:
            self.rpc_client.wait_for_load_embedding()

    def wait_for_dump_embedding(self):
        """Wait for dump the sparse embedding"""
        self.rpc_client.wait_for_dump_embedding()

    def wait_for_load_embedding(self):
        """Wait for load the sparse embedding"""
        self.rpc_client.wait_for_load_embedding()


def init_backend(
    worker_size: int = 20,
    replica_info: PyPersiaReplicaInfo = PyPersiaReplicaInfo(1, 0),
) -> Backend:
    """Initialize singleton Backend instance

    Arguments:
        worker_size (int, optional): rpc client thread pool size
        replica_info (PyPersiaReplicaInfo, optional): replica info of current process.
    """
    global _backend
    if not _backend:
        _backend = Backend(worker_size, replica_info)
    return _backend


def get_backend() -> Backend:
    """Get singleton Backend instance, raise PersiaRuntimeException once uninitialized the backend"""
    if not _backend:
        raise PersiaRuntimeException("init persia backend first")
    return _backend
