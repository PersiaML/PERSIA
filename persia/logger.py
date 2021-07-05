import logging

from colorlog import ColoredFormatter


class levelFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
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
    enable_err_redirect: bool = False,
    err_redirect_filepath: str = "error.log",
    err_redirect_level: int = logging.INFO,
) -> logging.Logger:
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
    return logging.getLogger(name)


def set_default_logger(name: str = None, **kwargs) -> logging.Logger:
    global _default_logger
    if not _default_logger:
        _default_logger = setLogger(name, **kwargs)
    return _default_logger


def get_default_logger(name: str = None, **kwargs) -> logging.Logger:
    """
    lazy init logger, use for global output
    """
    if _default_logger is None:
        set_default_logger(name or DEFAULT_LOGGER_NAME, **kwargs)
    return _default_logger


if __name__ == "__main__":

    logger = logging.getLogger("test")
    logger.debug("test logger")
    logger.info("test logger")
    logger.warning("test logger")
    logger.error("test logger")
    logger.critical("test logger")

    setLogger("test")

    logger.debug("test logger")
    logger.info("test logger")
    logger.warning("test logger")
    logger.error("test logger")
    logger.critical("test logger")
