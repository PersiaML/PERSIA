import os

from persia.service import get_embedding_worker_services


def test_get_embedding_worker_services():
    default_service = "embedding_worker:8887"
    for persia_service, service in zip(
        get_embedding_worker_services(), [default_service]
    ):
        assert persia_service == service

    embedding_worker_service = "localhost:8887"

    os.environ["EMBEDDING_WORKER_SERVICE"] = embedding_worker_service

    for persia_service, service in zip(
        get_embedding_worker_services(), [embedding_worker_service]
    ):
        assert persia_service == service
