import os

from typing import List


def get_embedding_worker_services() -> List[str]:
    """Get a list of current embedding worker services"""
    return (
        [os.environ["embedding_worker_service"]]
        if os.environ.get("embedding_worker_service", None)
        else ["embedding_worker:8887"]
    )
