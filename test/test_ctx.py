import os

from persia.helper import ensure_persia_service
from persia.ctx import BaseCtx, DataCtx

EMBEDDING_CONFIG = {"slots_config": {"age": {"dim": 8}}}
GLOBAL_CONFIG = {
    "embedding_worker_config": {"forward_buffer_size": 1000},
    "common_config": {"metrics_config": {"enable_metrics": False}},
}


def test_data_ctx():
    def data_loader():
        import numpy as np

        from persia.ctx import DataCtx
        from persia.embedding.data import PersiaBatch, IDTypeFeature, Label

        persia_batch = PersiaBatch(
            [
                IDTypeFeature(
                    "age",
                    [
                        np.array(
                            [
                                1,
                                2,
                                3,
                            ],
                            dtype=np.uint64,
                        )
                    ],
                )
            ],
            labels=[
                Label(
                    np.array(
                        [
                            1,
                        ],
                        dtype=np.float32,
                    )
                )
            ],
            requires_grad=False,
        )

        with DataCtx() as data_ctx:
            data_ctx.send_data(persia_batch)

    os.environ["WORLD_SIZE"] = str(1)
    os.environ["RANK"] = str(0)
    os.environ["LOCAL_RANK"] = str(0)

    from persia.ctx import BaseCtx
    from persia.data import Dataloder, StreamingDataset
    from persia.embedding import get_default_embedding_config
    from persia.env import get_world_size

    with ensure_persia_service(
        data_loader_func=data_loader,
        embedding_config=EMBEDDING_CONFIG,
        global_config=GLOBAL_CONFIG,
    ):
        embedding_config = get_default_embedding_config()

        with BaseCtx() as ctx:
            ctx.common_context.init_nats_publisher(get_world_size())
            ctx.common_context.configure_embedding_parameter_servers(
                embedding_config.emb_initialization[0],
                embedding_config.emb_initialization[1],
                embedding_config.admit_probability,
                embedding_config.weight_bound > 0,
                embedding_config.weight_bound,
            )
            ctx.common_context.wait_servers_ready()

            data_loader = Dataloder(
                StreamingDataset(buffer_size=10), timeout_ms=1000 * 15
            )
            data_generator = iter(data_loader)
            _ = next(data_generator)
