from typing import Optional, Union, List

import numpy as np

from persia.env import PERSIA_SKIP_CHECK_DATA
from persia.logger import get_default_logger
from persia.prelude import _PersiaBatch


_logger = get_default_logger()


# Maximum batch_size supported.
MAX_BATCH_SIZE = 65535

_ND_ARRAY_SUPPORT_TYPE = set(
    [np.bool_, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.uint8]
)


def _id_type_data_check(id_type_feature: np.ndarray, feature_name: str):
    """Check the type, dimension and dtype for id_type_feature.

    Arguments:
        id_type_feature (np.ndarray): id_type_feature array.
        feature_name (str): name of id_type_feature
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
    r"""Check if the dtype, shape and batch_size is valid or not.

    Arguments:
        data (np.ndarray): data that needs to be checked.
        data_name (str): name of data.
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
    """Check if batch size is equal to target_batch_size and no larger than to MAX_BATCH_SIZE."""
    assert (
        batch_size == target_batch_size
    ), f"expected {data_type}: {name} batch_size equal to {target_batch_size} but got {batch_size}"
    assert (
        batch_size <= MAX_BATCH_SIZE
    ), f"expected {data_type}:{name} batch_size <= MAX_BATCH_SIZE: {MAX_BATCH_SIZE} but got {batch_size}"


class IDTypeFeature:
    """:class:`IDTypeFeature` is a sparse matrix in `LIL <https://scipy-lectures.org/advanced/scipy_sparse/lil_matrix.html>`_
    format which contains categorical ID data.

    Example for :class:`IDTypeFeature`:

    .. code-block:: python

        import numpy as np
        from persia.embedding.data import IDTypeFeature

        lil_matrix = [
            np.array([1], dtype=np.uint64) for i in range(5)
        ]
        id_type_feature = IDTypeFeature("id_type_feature_with_single_id", lil_matrix)

        lil_matrix = [
            np.array([], dtype=np.uint64), # allow empty sample
            np.array([10001], dtype=np.uint64),
            np.array([], dtype=np.uint64),
            np.array([10002], dtype=np.uint64),
            np.array([10010], dtype=np.uint64)
        ]
        id_type_feature = IDTypeFeature("id_type_feature_with_empty_sample", lil_matrix)

    .. note::
        ``IDTypeFeature`` requires ``np.uint64`` as type for its elements.
    """

    def __init__(self, name: str, data: List[np.ndarray]):
        """
        Arguments:
            name (str): name of :class:`IDTypeFeature`.
            data (List[np.ndarray]): A lil sparse matrix data. Requires np.uint64 as
                type for its elements.
        """
        if not PERSIA_SKIP_CHECK_DATA:
            (_id_type_data_check(x, name) for x in data)

        self.name = name
        self.data = data

    @property
    def batch_size(self):
        return len(self.data)


class IDTypeFeatureWithSingleID:
    """The :class:`IDTypeFeatureWithSingleID` is a special format of :class:`IDTypeFeature` where there
    is only one id for each sample in the batch. :class:`IDTypeFeatureWithSingleID` only run a
    one-time type check compared to :class:`IDTypeFeature`. It can speed up the data
    preprocessing significantly with large batch size.

    Example for :class:`IDTypeFeatureWithSingleID`:

    .. code-block:: python

        import numpy as np
        from persia.embedding.data import IDTypeFeatureWithSingleID

        batch_with_single_id = np.array(
            [10001, 10002, 10010, 10020, 10030], np.uint64
        )
        id_type_feature = IDTypeFeatureWithSingleID("id_type_feature_demo", batch_with_single_id)

    .. note::
        ``IDTypeFeatureWithSingleID`` requires ``np.uint64`` as type for its elements.

    .. note::
        ``IDTypeFeatureWithSingleID`` does not allow empty sample in batch data. You
        should use ``IDTypeFeature`` instead in this case. See IDTypeFeature for more details.
    """

    def __init__(self, name: str, data: np.ndarray):
        """
        Arguments:
            name (str): name of :class:`IDTypeFeatureWithSingleID`.
            data (np.ndarray): :class:`IDTypeFeatureWithSingleID` data. Requires np.uint64 as type for
                its elements.
        """
        if not PERSIA_SKIP_CHECK_DATA:
            _id_type_data_check(data, name)

        self.name = name
        self.data = data

    @property
    def batch_size(self) -> int:
        return len(self.data)


class NdarrayDataBase:
    """The :class:`NdarrayDataBase` is a data structure that supports various datatypes and
    multi-dimensional data. PERSIA needs to convert the :class:`NdarrayDataBase` to the
    ``torch.Tensor`` so the datatype that it supports is the intersection of `NumPy
    datatype <https://numpy.org/doc/stable/user/basics.types.html#array-types-and-conversions-between-types>`_
    and `PyTorch datatype <https://pytorch.org/docs/stable/tensors.html#data-types>`_.

    Following datatype is supported for :class:`NdarrayDataBase`:

    +----------+
    |datatype  |
    +==========+
    |np.bool   |
    +----------+
    |np.int8   |
    +----------+
    |np.int16  |
    +----------+
    |np.int32  |
    +----------+
    |np.int64  |
    +----------+
    |np.float32|
    +----------+
    |np.float64|
    +----------+
    |np.uint8  |
    +----------+

    """

    DEFAULT_NAME = "ndarray_base"

    def __init__(self, data: np.ndarray, name: Optional[str] = None):
        """
        Arguments:
            data (np.ndarray): Numpy array.
            name (str, optional): name of data.
        """
        self.data = data
        self._name = name

        if not PERSIA_SKIP_CHECK_DATA:
            _ndarray_check(self.data, name)

    @property
    def batch_size(self) -> int:
        return self.data.shape[0]

    @property
    def name(self):
        return self._name or self.DEFAULT_NAME

    def __len__(self):
        return len(self.data)


class Label(NdarrayDataBase):

    """:class:`Label` is the ``subclass`` of :class:`NdarrayDataBase` that you can add various
    datatype and multi-dimensional data.

    Example for :class:`Label`:

    .. code-block:: python

        import numpy as np
        from persia.embedding.data import Label

        label_data = np.array([35000, 36000, 100000, 5000, 10000], dtype=np.float32)
        label = Label(label_data, name="income_label")

        label_data = np.array([True, False, True, False, True], dtype=np.bool)
        label = Label(label_data, name="ctr_bool_label")

    Or you can add multi-dimensional data to avoid memory fragments and type checks.

    .. code-block:: python

        import numpy as np
        from persia.embedding.data import Label

        label_data = np.array([
                [True, False],
                [False, True],
                [True, True],
                [False, False],
                [False, True]
            ], dtype=np.bool
        )
        label = Label(label_data, "click_with_is_adult")
    """

    DEFAULT_NAME = "label_anonymous"


class NonIDTypeFeature(NdarrayDataBase):
    """The :class:`NonIDTypeFeature` is the ``subclass`` of :class:`NdarrayDataBase` that you can add
    various datatypes and multi-dimensional data.

    Example for :class:`NonIDTypeFeature`:

    .. code-block:: python

        import numpy as np
        from persia.embedding.data import NonIDTypeFeature

        # float32 data
        non_id_type_feature_data = np.array([163, 183, 161, 190 ,170], dtype=np.float32)
        non_id_type_feature = NonIDTypeFeature(non_id_type_feature_data, "height")

        # image data
        non_id_type_feature_data = np.zeros((5, 3, 32, 32), dtype=np.uint8)
        non_id_type_feature = NonIDTypeFeature(non_id_type_feature_data, "image_data")
    """

    DEFAULT_NAME = "non_id_type_feature_anonymous"


class PersiaBatch:
    r"""The :class:`PersiaBatch` is the type of dataset used internally in Persia.
    It wraps the :class:`IDTypeFeature`, :class:`NonIDTypeFeature`, :class:`Label` and meta bytes data.

    Example for :class:`PersiaBatch`:

    .. code-block:: python

        import time
        import json
        import numpy as np
        from persia.embedding.data import PersiaBatch, NonIDTypeFeature, IDTypeFeature, Label

        batch_size = 1024

        non_id_type_feature = NonIDTypeFeature(np.zeros((batch_size, 2), dtype=np.float32))

        label = Label(np.ones((batch_size, 2), dtype=np.float32))

        id_type_feature_num = 3
        id_type_feature_max_sample_length = 100
        id_type_features = [
            IDTypeFeature(f"feature_{idx}",
                    [
                        np.ones(
                            (np.random.randint(id_type_feature_max_sample_length)),
                            dtype=np.uint64
                        )
                        for _ in range(batch_size)
                    ]
            ) for idx in range(id_type_feature_num))
        ]

        meta_info = {
            timestamp: time.time(),
            weight: 0.9,
        }
        meta_bytes = json.dumps(meta_info)

        persia_batch = PersiaBatch(id_type_features,
            non_id_type_features=[non_id_type_feature],
            labels=[label] requires_grad=requires_grad,
            requires_grad=True
            meta=meta_bytes
        )

    .. note::
        :class:`Label` data should be exists if set ``requires_grad=True``.
    """

    def __init__(
        self,
        id_type_features: List[Union[IDTypeFeature, IDTypeFeatureWithSingleID]],
        non_id_type_features: Optional[List[NonIDTypeFeature]] = None,
        labels: Optional[List[Label]] = None,
        batch_size: Optional[int] = None,
        requires_grad: bool = True,
        meta: Optional[bytes] = None,
    ):
        """
        Arguments:
            id_type_features (List[Union[IDTypeFeatureWithSingleID, IDTypeFeature]]):
                categorical data whose datatype should be uint64.
            non_id_type_features (List[NonIDTypeFeature], optional): dense data.
            labels (List[Label], optional): labels data.
            batch_size (int, optional): number of samples in each batch. :class:`IDTypeFeature`,
                :class:`NonIDTypeFeature` and :class:`Label` should have the same batch_size.
            requires_grad (bool, optional): set requires_grad for id_type_features.
            meta (bytes, optional): binary data.
        """

        assert len(id_type_features) > 0, "id_type_features should not be empty"
        batch_size = batch_size or id_type_features[0].batch_size

        self.batch = _PersiaBatch()
        for id_type_feature in id_type_features:
            _batch_size_check(
                id_type_feature.batch_size,
                batch_size,
                "id_type_feature",
                id_type_feature.name,
            )

            if isinstance(id_type_feature, IDTypeFeatureWithSingleID):
                self.batch.add_id_type_feature_with_single_id(
                    id_type_feature.data, id_type_feature.name
                )
            elif isinstance(id_type_feature, IDTypeFeature):
                self.batch.add_id_type_feature(
                    id_type_feature.data, id_type_feature.name
                )
            else:
                raise TypeError(
                    f"expected type of id_type_feature to be Union[IDTypeFeatureWithSingleID, IDTypeFeature] but got {type(id_type_feature)}"
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
        self.batch.converted_id_type_features2embedding_tensor(requires_grad)

    @property
    def data(self) -> _PersiaBatch:
        return self.batch

    def to_bytes(self) -> bytes:
        """Serialize persia_batch to bytes after checking."""
        return self.data.to_bytes()
