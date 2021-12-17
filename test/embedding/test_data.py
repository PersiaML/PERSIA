import numpy as np

import pytest

from persia.embedding.data import (
    _ND_ARRAY_SUPPORT_TYPE,
    NdarrayDataBase,
    IDTypeFeature,
    IDTypeFeatureWithSingleID,
    PersiaBatch,
)
from persia.prelude import check_pyarray_dtype_valid


def test_ndarray_base_dtype_convert_to_tensor():
    for dtype in _ND_ARRAY_SUPPORT_TYPE:
        data = np.zeros((1), dtype=dtype)
        check_pyarray_dtype_valid(data, data.dtype)


def test_ndarray_base_data():
    # test dtype support
    for dtype in _ND_ARRAY_SUPPORT_TYPE:
        NdarrayDataBase(np.zeros(1, dtype=dtype))

    # test batch_size and name
    ndarray_base_name = "test_name"
    data = NdarrayDataBase(np.array([1]), ndarray_base_name)
    assert data.batch_size == 1
    assert data.name == ndarray_base_name


def test_id_type_feature():
    id_type_feature_name = "test_name"

    assert (
        IDTypeFeature(id_type_feature_name, [np.array([1], dtype=np.uint64)]).batch_size
        == 1
    )


def test_sparse_id_type_feature():
    id_type_feature_name = "test_name"

    assert (
        IDTypeFeatureWithSingleID(
            id_type_feature_name, np.array([1], dtype=np.uint64)
        ).batch_size
        == 1
    )


def test_persia_batch():

    # test grad without label
    with pytest.raises(RuntimeError):
        PersiaBatch(
            id_type_features=[
                IDTypeFeature("test_name", [np.array([1], dtype=np.uint64)])
            ]
        )

    # test serialize bytes
    persia_batch = PersiaBatch(
        id_type_features=[IDTypeFeature("test_name", [np.array([1], dtype=np.uint64)])],
        requires_grad=False,
    )

    persia_batch_bytes = persia_batch.to_bytes()
    assert isinstance(persia_batch_bytes, bytes)
