import os
import time
import yaml

from persia.error import FileNotFoundException


def load_yaml(filepath: str) -> dict:
    """load the yaml config by provided filepath

    Arguments:
        filepath (str): yaml config path
    """
    if os.path.exists(filepath):
        raise FileNotFoundException(f"filepath {filepath} not found!")

    with open(filepath, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def block(interval: int = 120):
    """block the process by sleep function

    Arguments:
        interval (int): sleep interval
    """
    while True:
        time.sleep(interval)
