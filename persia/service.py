import os

from typing import List


def get_middleware_services() -> List[str]:
    """return the current middleware service list by environment variable"""
    return (
        [os.environ["MIDDLEWARE_SERVICE"]]
        if os.environ.get("MIDDLEWARE_SERVICE", None)
        else ["middleware:8887"]
    )
