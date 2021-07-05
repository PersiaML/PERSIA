import os

from typing import List, Tuple


def get_middleware_services() -> List[str]:
    return (
        [os.environ["MIDDLEWARE_SERVICE"]]
        if os.environ.get("MIDDLEWARE_SERVICE", None)
        else ["middleware:8000"]
    )


def get_client_services() -> List[str]:
    return (
        [os.environ["TRAINER_SERVICE"]]
        if os.environ.get("TRAINER_SERVICE", None)
        else ["trainer:8000"]
    )
