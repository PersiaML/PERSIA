from typing import List, Tuple

from persia.prelude import (
    PyPersiaRpcClient,
    PyPersiaBatchData,
    PyPersiaMessageQueueClient,
)

from persia.service import get_middleware_services, get_client_services
from persia.error import PersiaRuntimeException

_backend = None


class Backend:
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
    if not _backend:
        raise PersiaRuntimeException("init persia backend first")
    return _backend
