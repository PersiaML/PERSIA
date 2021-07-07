from typing import List, Tuple

from persia.prelude import (
    PyPersiaRpcClient,
    PyPersiaBatchData,
    PyPersiaMessageQueueClient,
)

from persia.service import get_middleware_services, get_client_services
from persia.error import PersiaRuntimeException

_backend = None

import torch


class Backend:
    r"""PersiaRpcClient wrapper that provide invoke middleware rpc function

    Arguments:
        worker_size (int): rpc client thread pool size
        middleware_services (List[str]): middleware address
        wait_server_ready (bool): whether to wait server configuration ready
        output_addrs (List[str]): message queue addrs for generate the output message queue
        init_output (bool): whether init the output message queue
    """

    def __init__(
        self,
        worker_size: int,
        middleware_services: List[str],
        wait_server_ready: bool,
        output_addrs: List[str],
        init_output: bool = False,
    ):
        self._backend = PyPersiaRpcClient(
            middleware_services, wait_server_ready, worker_size
        )

        self.output_services = []
        self.iter_idx = 0

        if init_output:
            self.init_output(output_addrs)

    def init_output(self, output_addrs: List[str]):
        """init the rpc wrapper output message queue for data compose scence
        Arguments:
            output_addrs (List(str)): message queue addrs for generate the output message queue
        """
        for output_addr in output_addrs:
            self.output_services.append(PyPersiaMessageQueueClient(output_addr))

    def __getattribute__(self, name: str):
        # TODO: remove the current function after refactor
        # py_client
        return getattr(
            object.__getattribute__(self, "_backend"), name, None
        ) or object.__getattribute__(self, name)

    @property
    def output_client(self):
        client = self.output_services[self.iter_idx]
        self.iter_idx = (self.iter_idx + 1) % len(self.output_services)
        return client

    def send_data(self, data: PyPersiaBatchData):
        """send data from data compose to trainer side

        Arguments:
            data (PyPersiaBatchData): persia_batch_data
        """
        self._backend.forward_id(data)
        bytes_data = data.to_bytes()
        self.output_client.put(bytes_data)


def init_backend(
    worker_size: int = 20,
    middleware_services: List[str] = None,
    wait_server_ready: bool = False,
    output_addrs: List[str] = None,
    init_output: bool = False,
) -> PyPersiaRpcClient:
    """Initialize the rpc wrapper singleton instance

    Arguments:
        worker_size (int): rpc client thread pool size
        middleware_services (List(str)): middleware address
        wait_server_ready (bool): whether to wait server configuration ready
        output_addrs (List(str)): message queue addrs for generate the output message queue
        init_output: whether init the output message queue
    """
    global _backend
    if not _backend:
        # TODO: Add service auto retrive...
        if not middleware_services:
            middleware_services = get_middleware_services()

        output_addrs = output_addrs or get_client_services() if init_output else None
        _backend = Backend(
            worker_size,
            middleware_services,
            wait_server_ready,
            output_addrs,
            init_output,
        )
    return _backend


def get_backend():
    """get rpc wrapper instance"""
    if not _backend:
        raise PersiaRuntimeException("init persia backend first")
    return _backend
