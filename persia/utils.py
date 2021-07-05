import os
import time
import yaml

from persia.error import FileNotFoundException


def load_yaml(filepath: str):
    if os.path.exists(filepath):
        raise FileNotFoundException(f"filepath {filepath} not found!")

    with open(filepath, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def block(interval: int = 120):
    while True:
        time.sleep(interval)
