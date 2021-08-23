import sys

from types import ModuleType

# pytype: disable=import-error
import persia_core

# pytype: enable=import-error


def register_submodule(module: ModuleType, root_module_path: str):
    """register the persia core module to sys module path.

    Arguments:
        module (ModuleType): Root module.
        root_module_path (str): Root module path.
    """
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
    persia_core,
    persia_core.__name__,
)


# pytype: disable=import-error
from persia_core import (
    is_cuda_feature_available,
    init_persia_embedding_staleness_semaphore,
)

from persia_core import (
    PyPersiaRpcClient,
)
from persia_core.optim import PyOptimizerBase
from persia_core.data import (
    PyPersiaBatchData,
)
from persia_core.utils import (
    PyPersiaMessageQueueServer,
    PyPersiaMessageQueueClient,
    PyPersiaBatchDataChannel,
    PyPersiaBatchDataSender,
    PyPersiaBatchDataReceiver,
    PyPersiaReplicaInfo,
)
from persia_core.nats import PyPersiaBatchFlowNatsStubPublisher, init_responder

if is_cuda_feature_available():
    from persia_core.backward import PyBackward
    from persia_core.forward import (
        PyForward,
        forward_directly_from_data,
        forward_directly_from_bytes,
    )

# pytype: enable=import-error
