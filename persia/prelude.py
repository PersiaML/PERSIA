import sys

from types import ModuleType

# pytype: disable=import-error
import persia_embedding_py_client_sharded_server

# pytype: enable=import-error


def register_submodule(module: ModuleType, root_module_path: str):
    for attr in dir(module):
        if attr.startswith("__"):
            continue
        obj = getattr(module, attr)
        if isinstance(obj, ModuleType):
            submodule_name = attr
            full_path = f"{root_module_path}.{submodule_name}"
            sys.modules[full_path] = obj
            register_submodule(obj, full_path)


register_submodule(
    persia_embedding_py_client_sharded_server,
    persia_embedding_py_client_sharded_server.__name__,
)


# pytype: disable=import-error
from persia_embedding_py_client_sharded_server import (
    PyPersiaRpcClient,
)
from persia_embedding_py_client_sharded_server.optim import PyOptimizerBase
from persia_embedding_py_client_sharded_server.data import PyPersiaBatchData
from persia_embedding_py_client_sharded_server.utils import (
    PyPersiaMessageQueueServer,
    PyPersiaMessageQueueClient,
)
from persia_embedding_py_client_sharded_server.backward import PyBackward
from persia_embedding_py_client_sharded_server.forward import PyForward

# pytype: enable=import-error
