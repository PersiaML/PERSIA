from persia.helper import ensure_persia_service
from persia.ctx import DataCtx

embedding_config = {"slot_configs": [{}]}


def test_data_ctx():

    with ensure_persia_service():
        ...
