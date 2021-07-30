import os
import time
import yaml

from persia.error import FileNotFoundException


def setup_seed(seed: int):
    """Set the random seed for commonly used Python library to ensure the experiment reproducibility.

    Arguments:
        seed (int): random seed.
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
    """load the yaml config provided by filepath

    Arguments:
        filepath (str): yaml config path
    """
    if os.path.exists(filepath):
        raise FileNotFoundException(f"filepath {filepath} not found!")

    with open(filepath, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def block(interval: int = 120):
    """block the process for ``interval`` seconds.

    Arguments:
        interval (int, optinal): sleep interval
    """
    while True:
        time.sleep(interval)
