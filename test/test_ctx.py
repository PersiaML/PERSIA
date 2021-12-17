import os

from typing import List

import numpy as np
import torch
import pytest

from persia.helper import ensure_persia_service
from persia.ctx import BaseCtx, DataCtx

from .utils import random_port

EMBEDDING_CONFIG = {"slots_config": {"age": {"dim": 8}}}
RAW_EMBEDDING_CONFIG = {
    "slots_config": {
        "user_id": {"dim": 8},
        "user_id_follower_list": {"dim": 8, "embedding_summation": False},
    }
}

GLOBAL_CONFIG = {
    "embedding_worker_config": {"forward_buffer_size": 1000},
    "common_config": {"metrics_config": {"enable_metrics": False}},
}


def assert_ndarray_base_data(
    ndarray_base_data_list: List[np.ndarray],
    tensors: List[torch.Tensor],
    use_cuda: bool,
):
    assert len(ndarray_base_data_list) == len(tensors)
    for ndarray_base_data, tensor in zip(ndarray_base_data_list, tensors):
        if use_cuda:
            tensor = tensor.cpu()

        np.testing.assert_equal(ndarray_base_data, tensor.numpy())


def assert_id_type_feature_data(tensors: List[torch.Tensor], config: dict):
    embedding_configs = config["slots_config"]
    for tensor, embedding_config in zip(tensors, embedding_configs):
        expected_dim = (
            embedding_config["dim"]
            if embedding_config.get("embedding_summation", True)
            else embedding_config["dim"] + 1
        )
        expected_ndim = 2 if embedding_config["embedding_summation"] else 3
        assert len(tensor.shape) == expected_ndim
        assert tensor.shape[-1] == expected_dim


# FIXME: Try no-singleton PersiaCommonContext.
# Every time init the PersiaCommonContext, it will reuse the instance created
# before. Any environment update makes no effects on singleton instance PersiaCommonContext,
# such as PERSIA_NATS_URL.

if torch.cuda.is_available():
    parameter_list = [True]
    ids = ["cuda"]
else:
    parameter_list = [False]
    ids = ["cpu"]


@pytest.mark.parametrize("use_cuda", parameter_list, ids=ids)
def test_data_ctx(use_cuda: bool):
    non_id_type_features = [np.array([1], dtype=np.float32)]
    labels = [
        np.array(
            [
                1,
            ],
            dtype=np.float32,
        )
    ]

    def data_loader():
        from persia.embedding.data import (
            PersiaBatch,
            IDTypeFeature,
            NonIDTypeFeature,
            Label,
        )

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
            non_id_type_features=[
                NonIDTypeFeature(non_id_type_feature)
                for non_id_type_feature in non_id_type_features
            ],
            labels=[Label(label) for label in labels],
            requires_grad=False,
        )

        with DataCtx() as data_ctx:
            data_ctx.send_data(persia_batch)

    os.environ["WORLD_SIZE"] = str(1)
    os.environ["RANK"] = str(0)
    os.environ["LOCAL_RANK"] = str(0)

    from persia.ctx import PreprocessMode, _prepare_feature
    from persia.data import DataLoader, StreamingDataset
    from persia.embedding import get_default_embedding_config
    from persia.env import get_world_size

    device_id = 0 if use_cuda else None

    with ensure_persia_service(
        data_loader_func=data_loader,
        embedding_config=EMBEDDING_CONFIG,
        global_config=GLOBAL_CONFIG,
        embedding_worker_port=random_port(),
        embedding_parameter_server_port=random_port(),
        nats_server_port=random_port(),
    ):
        embedding_config = get_default_embedding_config()

        with BaseCtx(device_id=device_id) as ctx:
            ctx.common_context.init_nats_publisher(get_world_size())
            ctx.common_context.configure_embedding_parameter_servers(
                embedding_config.emb_initialization[0],
                embedding_config.emb_initialization[1],
                embedding_config.admit_probability,
                embedding_config.weight_bound > 0,
                embedding_config.weight_bound,
            )
            ctx.common_context.wait_servers_ready()

            data_loader = DataLoader(
                StreamingDataset(buffer_size=10), timeout_ms=1000 * 30
            )
            data_generator = iter(data_loader)
            persia_training_batch = next(data_generator)
            (
                non_id_type_tensors,
                id_type_embedding_tensors,
                label_tensors,
            ) = _prepare_feature(persia_training_batch, PreprocessMode.EVAL)

            assert_ndarray_base_data(
                non_id_type_features, non_id_type_tensors, use_cuda
            )
            assert_ndarray_base_data(labels, label_tensors, use_cuda)
            # assert_id_type_feature_data(id_type_embedding_tensors, EMBEDDING_CONFIG)
