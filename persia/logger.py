import logging

from typing import Optional

from colorlog import ColoredFormatter


class levelFilter(logging.Filter):
    r"""log level filter to ensure log fileter

    Arguments:
        level (int): filter log level, remain the log greater than the setting level
    """

    def __init__(self, level: int):
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        """filter the record that level greater than the setting log level

        Arguments:
            record (logging.LogRecord): callback function input record items
        """
        return record.levelno > self.level


STREAM_LOG_FORMAT = "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(blue)s[%(filename)s:%(lineno)d]%(reset)s %(log_color)s%(message)s"
FILE_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
DEFAULT_LOGGER_NAME = "log"
_default_logger = None

LOG_COLOR = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}

COLOR_FORMATTER = ColoredFormatter(
    STREAM_LOG_FORMAT,
    datefmt=None,
    reset=True,
    log_colors=LOG_COLOR,
    secondary_log_colors={},
    style="%",
)
FORMATTER = logging.Formatter(
    FILE_LOG_FORMAT,
    datefmt=None,
    style="%",
)


def setLogger(
    name: str,
    log_level: int = logging.DEBUG,
    log_filename: str = "train.log",
    enable_file_logger: bool = False,
    err_redirect_filepath: str = "error.log",
    enable_err_redirect: bool = False,
    err_redirect_level: int = logging.INFO,
) -> logging.Logger:
    r"""this function make logger configuration simplify, provide the
    log_level and log filename.It also make error log redirect available

    Arguments:
        name (str): logger name
        log_filename (str): log filename
        enable_file_logger (bool): whether enable save log into file
        err_redirect_filepath (str): err log redirect filepath
        enable_err_redirect (bool): whether enable err log redirect
        err_redirect_level (int): error redirect log level
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(COLOR_FORMATTER)

    logger.addHandler(handler)
    logger.setLevel(log_level)

    if enable_file_logger:
        file_normal_handler = logging.FileHandler(log_filename, mode="a")
        file_normal_handler.setFormatter(FORMATTER)
        logger.addHandler(file_normal_handler)

    if enable_err_redirect:
        file_error_handler = logging.FileHandler(err_redirect_filepath, mode="a")
        file_error_handler.setFormatter(FORMATTER)
        file_error_handler.addFilter(levelFilter(err_redirect_level))
        logger.addHandler(file_error_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    r"""get logger by name

    Arguments:
        name (str): logger name
    """
    return logging.getLogger(name)


def _set_default_logger(name: str, **kwargs) -> logging.Logger:
    r"""set default logger

    Arguments:
        name (str): default logger name

    logging.Logger
    """
    global _default_logger
    if not _default_logger:
        _default_logger = setLogger(name, **kwargs)
    return _default_logger


def get_default_logger(name: Optional[str] = None, **kwargs) -> logging.Logger:
    r"""get default logger or init the default by given name

    Arguments:
        name (str, optional): logger name
    """
    if _default_logger is None:
        _set_default_logger(name or DEFAULT_LOGGER_NAME, **kwargs)
    return _default_logger
