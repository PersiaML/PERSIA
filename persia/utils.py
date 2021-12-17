import os
import yaml
import subprocess

from typing import List, Optional

from persia.logger import get_default_logger
from persia.env import PERSIA_LAUNCHER_VERBOSE

_logger = get_default_logger()


def setup_seed(seed: int):
    """Set the random seed for dependencies to ensure that experiments are reproducible.

    Arguments:
        seed (int): integer to use as seed for random number generator used by random,
            NumPy and pyTorch.
    """
    import numpy as np
    import torch
    import random

    np.random.seed(seed)

    random.seed(seed)

    torch.random.manual_seed(seed)
    if getattr(torch, "use_deterministic_algorithms", None):
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = True


def load_yaml(filepath: str) -> dict:
    """Load the yaml config by provided filepath.

    Arguments:
        filepath (str): yaml config path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"filepath {filepath} not found!")

    with open(filepath, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def dump_yaml(content: dict, filepath: str):
    """Dump the content into filepath."""

    with open(filepath, "w") as file:
        file.write(yaml.dump(content))


def run_command(cmd: List[str], env: Optional[dict] = None):
    cmd = list(map(str, cmd))
    if PERSIA_LAUNCHER_VERBOSE:
        cmd_str = " ".join(cmd)
        _logger.info(f"execute command: {cmd_str}")

    subprocess.check_call(cmd, env=env)


def resolve_binary_execute_path(binary_name: str) -> str:
    """Resolved executable file under persia package root."""
    return os.path.realpath(os.path.join(__file__, "../", binary_name))


def _is_port_available(port: int):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


MAXIMUM_LOCAL_PORT_NUM: int = 65535


def find_free_port(port: int = 10000, interval: int = 1) -> int:
    """Check current input port is available or not. It will add the interval to input port utils the
    the new port is available."""

    while not _is_port_available(port):
        port += interval
        if port > MAXIMUM_LOCAL_PORT_NUM:
            raise ValueError("free port not found.")
    return port
