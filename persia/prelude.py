import sys

from types import ModuleType

# pytype: disable=import-error
import persia_core

# pytype: enable=import-error


def register_submodule(module: ModuleType, root_module_path: str):
    """Register the persia core module to sys module path.

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
from persia_core import PersiaCommonContext, is_cuda_feature_available  # noqa

from persia_core.optim import OptimizerBase
from persia_core.data import PersiaBatch as _PersiaBatch
from persia_core.utils import (
    PersiaMessageQueueServer,
    PersiaMessageQueueClient,
    PersiaBatchDataChannel,
    PersiaBatchDataSender,
    PersiaBatchDataReceiver,
)
from persia_core.nats import init_responder  # noqa

from persia_core.backward import Backward  # noqa
from persia_core.forward import Forward, Tensor, PersiaTrainingBatch  # noqa

# pytype: enable=import-error
