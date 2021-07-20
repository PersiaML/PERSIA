import traceback

from contextlib import contextmanager

from persia.utils import block
from persia.logger import get_default_logger

_logger = get_default_logger()


@contextmanager
def persia_ctx(
    exit_when_meet_exception: bool = False,
    catch_exception: bool = True,
    verbose: bool = True,
):
    r"""Return a context warpper to process the error, to prevent process exit in docker service

    Arguments:
        exit_when_meet_exception (bool, optional): whether block the process when exit the contxt
        catch_exception (bool, optinal): catch the exception or not when occur the exception
        verbose (bool, optional): output the exception
    """
    try:
        yield
    except Exception as e:
        if verbose:
            exception_str = traceback.format_exc()
            _logger.error("\n" + exception_str)

        if not catch_exception:
            raise e

    if not exit_when_meet_exception:
        block()
