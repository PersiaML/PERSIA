import os

from typing import List


def get_embedding_worker_services() -> List[str]:
    """Get a list of current embedding worker services."""
    return (
        [os.environ["EMBEDDING_WORKER_SERVICE"]]
        if os.environ.get("EMBEDDING_WORKER_SERVICE", None)
        else ["embedding_worker:8887"]
    )
