import torch
import traceback

from contextlib import contextmanager

from persia.utils import block
from persia.logger import get_default_logger

logger = get_default_logger()


@contextmanager
def persia_ctx(
    exit_when_meet_exception: bool = False,
    catch_exception: bool = True,
    verbose: bool = True,
):
    try:
        yield
    except Exception as e:
        if verbose:
            exception_str = traceback.format_exc()
            logger.error("\n" + exception_str)

        if not catch_exception:
            raise e

    if not exit_when_meet_exception:
        block()
