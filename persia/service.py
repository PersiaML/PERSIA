import os

from typing import List


def get_middleware_services() -> List[str]:
    """Get a list of current middleware services"""
    return (
        [os.environ["MIDDLEWARE_SERVICE"]]
        if os.environ.get("MIDDLEWARE_SERVICE", None)
        else ["middleware:8887"]
    )
