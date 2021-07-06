from typing import List, Tuple
import time

from persia.prelude import (
    PyPersiaRpcClient,
    PyPersiaBatchData,
    PyPersiaReplicaInfo,
    PyPersiaBatchFlowNatsStubPublisher,
)
from persia.logger import get_default_logger
from persia.service import get_middleware_services, get_client_services
from persia.error import PersiaRuntimeException

_backend = None

logger = get_default_logger()

class Backend:
    def __init__(
        self,
        worker_size: int,
        replica_info: PyPersiaReplicaInfo,
    ):
        self.rpc_client = PyPersiaRpcClient(worker_size)
        self.nats_publisher = PyPersiaBatchFlowNatsStubPublisher(replica_info)

    def send_data(self, data: PyPersiaBatchData):
        tries = 10
        for _ in range(tries):
            try:
                self.nats_publisher.send_sparse_to_middleware(data)
                self.nats_publisher.send_dense_to_trainer(data)
            except:
                logger.warn('failed to send batch, retrying')
                time.sleep(10)
            else:
                break

    def set_configuration(
        self,
        initialize_lower: float,
        initialize_upper: float,
        admit_probability: float,
        enable_weight_bound: bool,
        weight_bound: float,
    ):
        tries = 10
        for _ in range(tries):
            try:
                self.nats_publisher.configure_sharded_servers(
                    initialize_lower,
                    initialize_upper,
                    admit_probability,
                    enable_weight_bound,
                    weight_bound,
                )
            except:
                logger.warn('failed to config sharded server, retrying')
                time.sleep(10)
            else:
                break
                

def init_backend(
    worker_size: int = 20,
    replica_info: PyPersiaReplicaInfo = PyPersiaReplicaInfo.trainer(1, 0),
) -> Backend:
    global _backend
    if not _backend:
        # TODO: Add service auto retrive...
        _backend = Backend(
            worker_size, replica_info
        )
    return _backend


def get_backend():
    if not _backend:
        raise PersiaRuntimeException("init persia backend first")
    return _backend
