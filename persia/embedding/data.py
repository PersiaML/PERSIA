from typing import Optional, Tuple, List, NewType

import numpy as np

from persia.prelude import _PersiaBatch
from persia.logger import get_default_logger


_logger = get_default_logger()


IDTypeFeatureSparse = NewType("IDTypeFeatureSparse", Tuple[str, List[np.ndarray]])

# TODO: implement dense format id_type_feature.
# Only one element for every sample in the batch
# IDTypeFeature = NewType("IDTypeFeature", Tuple[str, np.ndarray])


class PersiaBatch:
    r"""`PersiaBatch` store the id_type_features, non_id_type_features, labels and meta bytes data。

    Example:
        >>> import time
        >>> import json
        >>> ...
        >>> import numpy as np
        >>> ...
        >>> from persia.embedding.data import PersiaBatch
        >>> ...
        >>> batch_size = 1024
        >>> non_id_type_feature = np.zeros((batch_size, 2), dtype=np.float32)
        >>> label = np.ones((batch_size, 2), dtype=np.float32)
        >>> id_type_feature_num = 3
        >>> id_type_feature_max_sample_size = 100
        >>> id_type_features = [
        ...    (f"feature_{idx}", [np.ones((np.random.randint(id_type_feature_max_sample_size)), dtype=np.uint64)
        ...        for _ in range(batch_size)
        ...    ], for idx in range(id_type_feature_num))
        ... ]
        >>> meta_info = {
        ...     timestamp: time.time(),
        ...     weight: 0.9,
        ... }
        >>> meta_bytes = json.dumps(meta_info)
        >>> requires_grad = True
        >>> persia_batch = PersiaBatch(id_type_features, requires_grad=requires_grad, meta=meta_bytes)
        >>> persia_batch.add_non_id_type_feature(non_id_type_feature)
        >>> persia_batch.add_label(label)
    """

    def __init__(
        self,
        id_type_features: List[IDTypeFeatureSparse],
        requires_grad: bool = True,
        meta: Optional[bytes] = None,
    ):
        """
        Arguments:
            id_type_features (List[IDTypeFeatureSparse]): Categorical data with feature_name which datatype should be uint64.
            requires_grad (bool, optional): Set requires_grad for id_type_features.
            meta (bytes, optional): Binary data.
        """

        assert len(id_type_features) > 0, "id_type_features should not be empty"
        batch_size_arr = [
            len(id_type_feature[1]) for id_type_feature in id_type_features
        ]
        batch_size = batch_size_arr[0]
        assert all(
            map(lambda x: x == batch_size, batch_size_arr)
        ), f"all id_type_features should have same batch_size, id_type_features batch_size array is {batch_size_arr}"

        self.batch_size = batch_size
        self.id_type_features = id_type_features
        self.requires_grad = requires_grad
        self.has_label = False

        self.batch = _PersiaBatch()
        self.batch.add_id_type_features(id_type_features, requires_grad)

        if meta is not None:
            if isinstance(meta, (bytes, bytearray)):
                self.batch.add_meta(meta)
            else:
                _logger.warn(
                    f"Expect PersiaBatch.meta type is bytes or bytearray, but got {type(meta)}"
                )

    @property
    def data(self) -> _PersiaBatch:
        return self.batch

    def is_valid(self):
        r"""Check PersiaBatch valid or not."""

        assert not self.requires_grad or (
            self.requires_grad and self.has_label
        ), "PersiaBatch format invalid, labels should not be empty while required grad set to True"

    def _check_numpy_array(self, data: np.ndarray, data_name: str):
        r"""Check the dtype, shape and batch_size is valid or not.

        Arguments:
            data (np.ndarray): Data that need to check.
            data_name (str): Name of data.
        """
        assert isinstance(
            data, np.ndarray
        ), f"TypeError: input data {data_name}, type: {type(data)} no match numpy ndarray "

        assert (
            data.ndim > 0
        ), f"{data_name} ndarray got ndim: {data.ndim} expect ndim greater than one"

        assert (
            data.shape[0] == self.batch_size
        ), f"{data_name} ndarray batch_size not match with id_type_feature batch_size {self.batch_size}"

    def add_non_id_type_feature(
        self,
        non_id_type_feature: np.ndarray,
        non_id_type_feature_name: Optional[str] = None,
    ):
        r"""Add the non_id_type_feature into `PersiaBatch`.

        Arguments:
            non_id_type_feature (np.ndarray): Numerical data should not be empty and corresponding batch_size should equal to id_type_feature
            non_id_type_feature_name (str, optional): Name of non_id_type_feature.
        """
        self._check_numpy_array(non_id_type_feature, "non_id_type_feature")
        self.batch.add_non_id_type_feature(
            non_id_type_feature, non_id_type_feature.dtype, non_id_type_feature_name
        )

    def add_label(self, label: np.ndarray, label_name: Optional[str] = None):
        r"""Add the label into `PersiaBatch`.

        Arguments:
            label (np.ndarray): Label data is a ndarray that ndim should greather than one and batch_size should equal to id_type_feature.
            label_name (str, optional): Name of label
        """
        self._check_numpy_array(label, "label")
        self.batch.add_label(label, label.dtype, label_name)
        self.has_label = True

    def to_bytes(self) -> bytes:
        """Serialize persia_batch to bytes after checking."""
        self.is_valid()
        return self.data.to_bytes()
