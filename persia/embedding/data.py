import os

from typing import Optional, Union, List

import numpy as np

from persia.prelude import _PersiaBatch
from persia.logger import get_default_logger


_logger = get_default_logger()


# Maximum batch_size support.
MAX_BATCH_SIZE = 65535

# Skip check for PERSIA data except batch_size check.
# Raise RuntimeError when data is not meet requirement such as
# type, dtype or shape not match.
SKIP_CHECK_PERSIA_DATA = bool(int(os.environ.get("SKIP_CHECK_PERSIA_DATA", "0")))

_ND_ARRAY_SUPPORT_TYPE = set(
    [np.bool, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.uint8]
)


def _id_type_data_check(id_type_feature: np.ndarray, feature_name: str):
    """Check the type, dimension and dtype for id_type_feature.

    Arguments:
        id_type_feature (np.ndarray): id_type_feature array.
        feature_name (str): Name of id_type_feature
    """
    assert isinstance(
        id_type_feature, np.ndarray
    ), f"expected id_type_feature: {feature_name} type to be np.ndarray but got tpye: {type(id_type_feature)}"
    assert (
        id_type_feature.ndim == 1
    ), f"expected id_type_feature: {feature_name} ndim equal to one but got ndim: {id_type_feature.ndim}"
    assert (
        id_type_feature.dtype == np.uint64
    ), f"expected id_type_feature: {feature_name} dtype as np.uint64 but got dtype: {id_type_feature.dtype}"


def _ndarray_check(data: np.ndarray, data_name) -> bool:
    r"""Check the dtype, shape and batch_size is valid or not.

    Arguments:
        data (np.ndarray): Data that need to check.
        data_name (str): Name of data.
    """
    assert isinstance(
        data, np.ndarray
    ), f"input data {data_name}, type: {type(data)} no match numpy ndarray "
    assert (
        data.dtype.type in _ND_ARRAY_SUPPORT_TYPE
    ), f"np.array only support dtype {_ND_ARRAY_SUPPORT_TYPE} but got {data_name} dtype {data.dtype}"
    assert (
        data.ndim > 0
    ), f"{data_name} ndarray got ndim: {data.ndim} expect ndim greater than one"


def _batch_size_check(
    batch_size: int, target_batch_size: int, data_type: str, name: str
):
    """Check batch size that equal to target_batch_size and small and equal than MAX_BATCH_SIZE"""
    assert (
        batch_size == target_batch_size
    ), f"expected {data_type}: {name} batch_size equal to {target_batch_size} but got {batch_size}"
    assert (
        batch_size <= MAX_BATCH_SIZE
    ), f"expected {data_type}:{name} batch_size <= MAX_BATCH_SIZE: {MAX_BATCH_SIZE} but got {batch_size}"


class IDTypeFeatureSparse:
    """IDTypeFeature sparse form"""

    def __init__(self, feature_name: str, id_type_feature: List[np.ndarray]):
        """
        Arguments:
            feature_name (str): Name of IDTypeFeature.
            id_type_feature (List[np.ndarray]): IDTypeFeature data.A Sparse matrix that represent list of list.Only accpet np.uint64 element.
        """
        if not SKIP_CHECK_PERSIA_DATA:
            (_id_type_data_check(x, feature_name) for x in id_type_feature)

        self.feature_name = feature_name
        self.id_type_feature = id_type_feature

    @property
    def batch_size(self):
        return len(self.id_type_feature)


class IDTypeFeature:
    """IDTypeFeature special form that only exists one id for every sample in the batch."""

    def __init__(self, feature_name: str, id_type_feature: np.ndarray):
        """
        Arguments:
            feature_name (str): Name of IDTypeFeature
            id_type_feature (np.ndarray): IDTypeFeature data.A Sparse matrix that represent list of list.Only accpet np.uint64 element.
        """
        if not SKIP_CHECK_PERSIA_DATA:
            _id_type_data_check(id_type_feature, feature_name)

        self.feature_name = feature_name
        self.id_type_feature = id_type_feature

    @property
    def batch_size(self) -> int:
        return len(np.id_type_feature)


class _NdarrayDataBase:
    CLASS_NAME = "ndarray_base"

    def __init__(self, data: np.ndarray, name: str = None):
        """
        Arguments:
            data (np.ndarray): Numpy array.
            name (str, optional): Name of data.
        """
        self.data = data
        self._name = name

        if not SKIP_CHECK_PERSIA_DATA:
            _ndarray_check(self.data, name)

    @property
    def batch_size(self) -> int:
        return self.data.shape[0]

    @property
    def name(self):
        return self._name or self.CLASS_NAME

    def __len__(self):
        return len(self.data)


class Label(_NdarrayDataBase):
    CLASS_NAME = "label_anonymous"


class NonIDTypeFeature(_NdarrayDataBase):
    CLASS_NAME = "non_id_type_feature_anonymous"


class PersiaBatch:
    r"""`PersiaBatch` store the id_type_features, non_id_type_features, labels and meta bytes data。

    Example:
        >>> import time
        >>> import json
        >>> ...
        >>> import numpy as np
        >>> ...
        >>> from persia.embedding.data import PersiaBatch, NonIDTypeFeature, IDTypeFeatureSparse, Label
        >>> ...
        >>> batch_size = 1024
        >>> non_id_type_feature = NonIDTypeFeature(np.zeros((batch_size, 2), dtype=np.float32))
        >>> label = Label(np.ones((batch_size, 2), dtype=np.float32))
        >>> id_type_feature_num = 3
        >>> id_type_feature_max_sample_size = 100
        >>> id_type_features = [
        ...    IDTypeFeatureSparse(f"feature_{idx}", [np.ones((np.random.randint(id_type_feature_max_sample_size)), dtype=np.uint64)
        ...        for _ in range(batch_size)
        ...    ]), for idx in range(id_type_feature_num))
        ... ]
        >>> meta_info = {
        ...     timestamp: time.time(),
        ...     weight: 0.9,
        ... }
        >>> meta_bytes = json.dumps(meta_info)
        >>> requires_grad = True
        >>> persia_batch = PersiaBatch(id_type_features,
        ... non_id_type_features=[non_id_type_feature],
        ... labels=[label] requires_grad=requires_grad,
        ... meta=meta_bytes
        ... )
    """

    def __init__(
        self,
        id_type_features: List[Union[IDTypeFeatureSparse, IDTypeFeature]],
        non_id_type_features: Optional[List[NonIDTypeFeature]] = None,
        labels: Optional[List[Label]] = None,
        batch_size: Optional[int] = None,
        requires_grad: bool = True,
        meta: Optional[bytes] = None,
    ):
        """
        Arguments:
            id_type_features (List[Union[IDTypeFeatureSparse, IDTypeFeature]]): Categorical data with feature_name which datatype should be uint64.
            non_id_type_features (List[NonIdTypeFeature], optional): dense data.
            labels: (List[Label], optional): labels data.
            batch_size (int, optional): Num of batch_size.IDTypeFeatures, NonIDTypeFeatures and Labels should have same batch_size.
            requires_grad (bool, optional): Set requires_grad for id_type_features.
            meta (bytes, optional): Binary data.
        """

        assert len(id_type_features) > 0, "id_type_features should not be empty"
        batch_size = batch_size or id_type_features[0].batch_size

        self.batch = _PersiaBatch()
        for id_type_feature in id_type_features:
            _batch_size_check(
                id_type_feature.batch_size,
                batch_size,
                "id_type_feature",
                id_type_feature.feature_name,
            )

            if isinstance(id_type_feature, IDTypeFeature):
                self.batch.add_id_type_feature(
                    id_type_feature.id_type_feature, id_type_feature.feature_name
                )
            elif isinstance(id_type_feature, IDTypeFeatureSparse):
                self.batch.add_id_type_feature_sparse(
                    id_type_feature.id_type_feature, id_type_feature.feature_name
                )
            else:
                raise TypeError(
                    f"expected type of id_type_feature to be Union[IDTypeFeatureSparse, IDTypeFeature] but got {type(id_type_feature)}"
                )

        if non_id_type_features is not None:
            for non_id_type_feature in non_id_type_features:
                _batch_size_check(
                    non_id_type_feature.batch_size,
                    batch_size,
                    "non_id_type_feature",
                    non_id_type_feature.name,
                )
                self.batch.add_non_id_type_feature(
                    non_id_type_feature.data,
                    non_id_type_feature.data.dtype,
                    non_id_type_feature.name,
                )

        if labels is not None:
            for label in labels:
                _batch_size_check(label.batch_size, batch_size, "label", label.name)
                self.batch.add_label(label.data, label.data.dtype, label.name)

        if meta is not None:
            if isinstance(meta, bytes):
                self.batch.add_meta(meta)
            else:
                _logger.warn(
                    f"expect PersiaBatch.meta type is bytes but got {type(meta)}"
                )

        self.batch_size = batch_size
        self.batch.check_batch(requires_grad)

    @property
    def data(self) -> _PersiaBatch:
        return self.batch

    def to_bytes(self) -> bytes:
        """Serialize persia_batch to bytes after checking."""
        return self.data.to_bytes()
