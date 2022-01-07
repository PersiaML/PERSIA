import os
import io
import socket

from enum import Enum
from queue import Queue
from typing import List, Tuple, Optional, Union

import torch

from persia import env
from persia.logger import get_default_logger
from persia.embedding.optim import Optimizer
from persia.embedding import EmbeddingConfig, get_default_embedding_config
from persia.embedding.data import PersiaBatch
from persia.prelude import (
    PersiaCommonContext,
    PersiaTrainingBatch,
    Tensor,
)
from persia.distributed import DistributedBaseOption, get_default_distributed_option

_CURRENT_CXT = None

_logger = get_default_logger()


def _check_finite(tensors: List[torch.Tensor]) -> bool:
    """Check if all tensors in the input list contain only finite elements.

    Arguments:
        tensors (List[torch.Tensor]): list of tensor to be checked.

    Returns:
        bool: ``True`` if all elements in ``tensors`` are finite or None.
    """
    return all([torch.isfinite(t).all() if t is not None else True for t in tensors])


def _cast_dlpack2torch_tensor(
    tensor: Tensor, requires_grad: bool = False
) -> torch.Tensor:
    """Convert the DLPack PythonCapsule to torch tensor.

    Arguments:
        Tensor (Tensor): tensor wrapper that contains dlpack information.
        requires_grad (bool, optional): whether current tensor requires grad or not.
    Returns: pytorch tensor
    """

    import torch.utils.dlpack as dlpack

    tensor = dlpack.from_dlpack(tensor.dlpack)
    tensor.requires_grad = requires_grad
    return tensor


class PreprocessMode(Enum):
    r"""Mode of preprocessing.

    Used by :meth:`.prepare_features` to generate features of different datatypes.

    When set to :attr:`.TRAIN`, :meth:`.prepare_features` will return a torch tensor with
    ``requires_grad`` attribute set to ``True``. When set to :attr:`.EVAL`,
    :meth:`.prepare_features` will return a torch tensor with ``requires_grad`` attribute
    set to ``False``. :attr:`.INFERENCE` behaves almost identical to :attr:`.EVAL`,
    except that :attr:`.INFERENCE` allows :class:`EmbeddingCtx` to process the :class:`PersiaTrainingBatch`
    without a target tensor.
    """
    TRAIN = 1
    EVAL = 2
    INFERENCE = 3


def _prepare_feature(
    persia_training_batch: PersiaTrainingBatch,
    preprocess_mode: PreprocessMode = PreprocessMode.TRAIN,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[List[torch.Tensor]]]:
    if preprocess_mode == PreprocessMode.INFERENCE:
        persia_training_batch.label_torch_tensors = None
    else:
        # pytype: disable=attribute-error
        persia_training_batch.label_tensors = (
            persia_training_batch.consume_all_label_tensors()
        )
        # pytype: enable=attribute-error
        persia_training_batch.label_torch_tensors = [
            _cast_dlpack2torch_tensor(label_tensor)
            for label_tensor in persia_training_batch.label_tensors
        ]

    is_training = preprocess_mode == PreprocessMode.TRAIN  # cache property

    # pytype: disable=attribute-error
    persia_training_batch.non_id_type_feature_tensors = (
        persia_training_batch.consume_all_non_id_type_feature_tensors()
    )
    # pytype: enable=attribute-error
    persia_training_batch.non_id_type_feature_torch_tensors = [
        _cast_dlpack2torch_tensor(non_id_type_feature_tensor)
        for non_id_type_feature_tensor in persia_training_batch.non_id_type_feature_tensors
    ]

    # pytype: disable=attribute-error
    persia_training_batch.id_type_feature_embedding_tensors = (
        persia_training_batch.consume_all_id_type_feature_embedding_tensors()
    )
    # pytype: enable=attribute-error

    persia_training_batch.emb_slots = []  # cache embedding to prevent tensor expired
    id_type_feature_embedding_cache_torch_tensors = (
        []
    )  # cache origin embedding for later backward procedure
    id_type_feature_embedding_torch_tensors = (
        []
    )  # id type tensos for later forward procedure

    for (
        id_type_feature_embedding_tensor
    ) in persia_training_batch.id_type_feature_embedding_tensors:
        if id_type_feature_embedding_tensor.is_raw_embedding():
            # no duplicate id in raw_id_tensor
            (
                raw_embedding,
                index,
                non_empty_index,
                sample_id_num,
            ) = id_type_feature_embedding_tensor.get_raw_embedding()

            persia_training_batch.emb_slots.append(
                [raw_embedding, index, non_empty_index]
            )
            distinct_id_tensor = _cast_dlpack2torch_tensor(raw_embedding)
            index_tensor = _cast_dlpack2torch_tensor(
                index
            )  # tensor shape (1, batch_size * sample_fixed_size)
            max_index = index_tensor.max()
            size_of_distinct_id_tensor = distinct_id_tensor.shape[0]

            assert (
                max_index < size_of_distinct_id_tensor
            ), "raw embedding select index larger than tensor"

            non_empty_index_tensor = _cast_dlpack2torch_tensor(
                non_empty_index
            )  # tensor shape (-1), variable length

            batch_size = len(sample_id_num)
            dim = distinct_id_tensor.shape[-1]
            sample_fixed_size = index_tensor.shape[-1] // batch_size
            index_select_raw_tensor = distinct_id_tensor.index_select(
                0, index_tensor.view(-1)
            )
            index_select_raw_tensor.requires_grad = is_training

            raw_fixed_size_tensor = index_select_raw_tensor.view(
                -1, sample_fixed_size, dim
            )
            mask = (
                index_tensor.view(batch_size, sample_fixed_size, 1) != 0
            ).half()  # generate mask
            raw_fixed_size_tensor_with_mask = torch.cat(
                [raw_fixed_size_tensor, mask], dim=2
            )
            id_type_feature_embedding_cache_torch_tensors.append(
                (
                    raw_embedding.name,
                    distinct_id_tensor,
                    index_tensor,
                    non_empty_index_tensor,
                    index_select_raw_tensor,
                )
            )
            id_type_feature_embedding_torch_tensors.append(
                raw_fixed_size_tensor_with_mask
            )
        else:
            embedding = id_type_feature_embedding_tensor.get_sum_embedding()
            persia_training_batch.emb_slots.append([embedding])
            attention_sum_tensor = _cast_dlpack2torch_tensor(
                embedding, requires_grad=is_training
            )
            id_type_feature_embedding_torch_tensors.append(attention_sum_tensor)
            id_type_feature_embedding_cache_torch_tensors.append(
                (embedding.name, None, None, None, attention_sum_tensor)
            )

    persia_training_batch.id_type_feature_embedding_torch_tensors = (
        id_type_feature_embedding_torch_tensors
    )
    persia_training_batch.id_type_feature_embedding_cache_torch_tensors = (
        id_type_feature_embedding_cache_torch_tensors
    )

    return (
        persia_training_batch.non_id_type_feature_torch_tensors,
        persia_training_batch.id_type_feature_embedding_torch_tensors,
        persia_training_batch.label_torch_tensors,
    )


class BaseCtx:
    r"""Initializes a common context for other persia context, e.g. :class:`DataCtx`,
    :class:`EmbeddingCtx` and :class:`TrainCtx`. This class should not be instantiated
    directly.
    """

    def __init__(
        self, threadpool_worker_size: int = 10, device_id: Optional[int] = None
    ):
        """
        Arguments:
            threadpool_worker_size (int): rpc threadpool worker size.
            device_id (int, optional): the CUDA device to use for this process.
        """
        self.origin_context = None

        if device_id is not None and device_id >= 0:
            assert torch.cuda.is_available() and (
                0 <= device_id < torch.cuda.device_count()
            ), f"device_id: {device_id} invalid!"

            torch.cuda.set_device(device_id)
        else:
            device_id = None

        self.device_id = device_id

        # PersiaCommonContext initialize with the rank and world size if
        # it can retrive corresponding information
        if env.get_rank() is not None:
            replica_index = env.get_rank()
            replica_size = env.get_world_size()
        else:
            replica_index = env.get_replica_index()
            replica_size = env.get_replica_size()

        self.common_context = PersiaCommonContext(
            threadpool_worker_size, replica_index, replica_size, device_id
        )
        _logger.info(
            f"init persia context, replica_size: {replica_size} replica_index: {replica_index}"
        )

    def _enter(self):
        """Hook when enter the context"""
        ...

    def _exit(self):
        """Hook when exit the context"""
        ...

    def __enter__(self):
        self._enter()

        global _CURRENT_CXT
        self.origin_context = _CURRENT_CXT
        _CURRENT_CXT = self

        return self

    def __exit__(self, exc_type, value, trace):
        self._exit()

        global _CURRENT_CXT
        _CURRENT_CXT = self.origin_context

        if exc_type:
            import traceback

            _logger.error("\n" + traceback.format_exc())


class DataCtx(BaseCtx):
    r"""Data context provides the communication functionality to data generator component.
    Used for sending a :class:`PersiaBatch` to the `nn worker` and `embedding worker`.

    If you use the :class:`DataCtx` to send the :class:`PersiaBatch` on `data-loader`, you
    should use the :class:`StreamingDataset` to receive the data on `nn-worker`.

    On `data-loader`:

    .. code-block:: python

        from persia.ctx import DataCtx
        from persia.embedding.data import PersiaBatch

        loader = make_loader()
        with DataCtx() as ctx:
            for (non_id_type_features, id_type_features, labels) in loader:
                batch_data = PersiaBatch(
                    id_type_features=id_type_features,
                    non_id_type_features,
                    label,
                    requires_grad=True
                )
                ctx.send_data(persia_batch)

    On `nn-worker`:

    .. code-block:: python

        from persia.ctx import TrainCtx
        from persia.data import StreamingDataset, DataLoader

        buffer_size = 15

        streaming_dataset = StreamingDataset(buffer_size)
        data_loader = DataLoader(streaming_dataset)

        with TrainCtx(...):
            for persia_training_batch in data_loader:
                ...

    .. note::
        The examples cannot be run directly, you should launch the `nn_worker`, `embedding-worker`,
        `embedding-parameter-server`, and `nats-server` to ensure the example gets the correct result.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(DataCtx, self).__init__(*args, **kwargs)
        self._prepare()
        _logger.info("Data ctx prepare done.")

    def _prepare(self):
        """Do some preparation to init `DataCtx`."""

        self.common_context.init_nats_publisher()
        self.common_context.wait_servers_ready()

    def send_data(self, persia_batch: PersiaBatch):
        """Send PersiaBatch from data loader to nn worker and embedding worker side.

        Arguments:
            persia_batch (PersiaBatch): :class:`PersiaBatch` that haven't been processed.
        """
        self.common_context.send_id_type_features_to_embedding_worker(persia_batch.data)
        self.common_context.send_non_id_type_features_to_nn_worker(persia_batch.data)


class EmbeddingCtx(BaseCtx):
    r"""Provides the embedding-related functionality. :class:`EmbeddingCtx` can run offline test
    or online inference depending on different preprocess_mode. The simplest way to get
    this context is by using :func:`eval_ctx` to get the
    :class:`EmbeddingCtx` instance.

    Example for :class:`EmbeddingCtx`:

    .. code-block:: python

        from persia.ctx import EmbeddingCtx, PreprocessMode
        from persia.embedding.data import PersiaBatch

        model = get_dnn_model()
        loader = make_dataloader()
        device_id = 0

        with EmbeddingCtx(
            PreprocessMode.EVAL,
            model=model,
            device_id=device_id
        ) as ctx:
            for (non_id_type_features, id_type_features, labels) in loader:
                persia_batch = PersiaBatch(
                    id_type_features
                    non_id_type_features=non_id_type_features,
                    labels=labels
                    requires_grad=False
                )
                persia_training_batch = ctx.get_embedding_from_data(persia_batch)
                (output, label) = ctx.forward(persia_training_batch)

    .. note::
        The examples cannot be run directly, you should launch the `nn_worker`,
        `embedding-worker`, `embedding-parameter-server`, and `nats-server` to
        ensure the example gets the correct result.

    .. note::
        If you set ``device_id=None``, the training data and the model will be placed
        in host memory rather than in `CUDA` device memory by default.

    """

    def __init__(
        self,
        preprocess_mode: PreprocessMode,
        model: Optional[torch.nn.Module] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        *args,
        **kwargs,
    ):
        """
        Arguments:
            preprocess_mode (PreprocessMode): different preprocess mode effect the
                behavior of :meth:`.prepare_features`.
            model (torch.nn.Module): denese neural network PyTorch model.
            embedding_config (EmbeddingConfig, optional): the embedding configuration that
                will be sent to the embedding server.
        """
        super(EmbeddingCtx, self).__init__(*args, **kwargs)
        self.preprocess_mode = preprocess_mode
        self.model = model
        self.embedding_config = embedding_config or get_default_embedding_config()

        self.current_batch = None

    def _enter(self):
        if self.embedding_config is not None:
            self.configure_embedding_parameter_servers(self.embedding_config)

    def configure_embedding_parameter_servers(
        self,
        embedding_config: EmbeddingConfig,
    ):
        """Apply :class:`EmbeddingConfig` to embedding servers.

        Arguments:
            embedding_config (EmbeddingConfig): the embedding configuration that will
                be sent to the embedding server.
        """
        self.common_context.configure_embedding_parameter_servers(
            embedding_config.emb_initialization[0],
            embedding_config.emb_initialization[1],
            embedding_config.admit_probability,
            embedding_config.weight_bound > 0,
            embedding_config.weight_bound,
        )

    def forward(
        self, batch: PersiaTrainingBatch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Call :meth:`.prepare_features` and then do a forward step of the model in context.

        Arguments:
            batch (PersiaTrainingBatch): training data provided by PERSIA upstream including
                non_id_type_features ,labels, id_type_feature_embeddings and meta info.

        Returns:
            the tuple of output data and target data.
        """
        assert self.model is not None, "model not found, please init context with model"
        non_id_type_tensors, embedding_tensors, labels = self.prepare_features(batch)
        output = self.model(non_id_type_tensors, embedding_tensors)
        return (output, labels)

    def prepare_features(
        self, persia_training_batch: PersiaTrainingBatch
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[List[torch.Tensor]]]:
        r"""This function converts data from ``PersiaTrainingBatch`` to ``torch.Tensor``.

        :class:`PersiaTrainingBatch` contains non_id_type_features, id_type_feature_embeddings
        and labels. But they can't use directly in training before convert the :class:`Tensor`
        to ``torch.Tensor``.

        Arguments:
            persia_training_batch (PersiaTrainingBatch): training data provided by PERSIA
                upstream including non_id_type_features, labels, id_type_feature_embeddings
                and meta info.

        Returns:
            the tuple of non_id_type_feature_tensors, id_type_feature_embedding_tensors and
            label_tensors.
        """
        self.current_batch = persia_training_batch
        return _prepare_feature(persia_training_batch, self.preprocess_mode)

    def dump_checkpoint(
        self,
        dst_dir: str,
        dense_filename: str = "dense.pt",
        jit_dense_filename: str = "jit_dense.pt",
        blocking: bool = True,
        with_jit_model: bool = False,
    ):
        """Save the model checkpoint (both dense and embedding) to the destination directory.

        Arguments:
            dst_dir (str): destination directory.
            dense_filename (str, optional): dense checkpoint filename.
            jit_dense_filename (str, optional): dense checkpoint filename after
                PyTorch jit script.
            blocking (bool, optional): dump embedding checkpoint in blocking mode or not.
            with_jit_model (bool, optional): dump jit script dense checkpoint or not.
        """
        assert self.model is not None, "model not found, please init context with model"

        if with_jit_model:
            self.dump_torch_state_dict(self.model, dst_dir, jit_dense_filename, True)
        self.dump_torch_state_dict(self.model, dst_dir, dense_filename)

        self.dump_embedding(dst_dir, blocking=blocking)

    def load_checkpoint(
        self,
        src_dir: str,
        map_location: Optional[str] = None,
        dense_filename: str = "dense.pt",
        blocking: bool = True,
    ):
        """Load the dense and embedding checkpoint from the source directory.

        Arguments:
            src_dir (str): source directory.
            map_location (str, optional): load the dense checkpoint to specific device.
            dense_filename (str, optional): dense checkpoint filename.
            blocking (bool, optional): dump embedding checkpoint in blocking mode or not.
        """
        assert self.model is not None, "model not found, please init context with model"

        dense_model_filepath = os.path.join(src_dir, dense_filename)
        if os.path.exists(dense_model_filepath):
            self.load_torch_state_dict(
                self.model, dense_model_filepath, map_location=map_location
            )

        self.load_embedding(src_dir, blocking=blocking)

    def dump_embedding(self, dst_dir: str, blocking: bool = True):
        """Dump embeddings to the destination directory.
        By default, this function is synchronous and will wait for the completion
        of embedding loading before returning. This is done internally through
        a call to :meth:`.wait_for_dump_embedding`.
        Set ``blocking=False`` to allow asyncronous computation,
        in which case the function will return immediately.
        :meth:`.wait_for_dump_embedding` to wait until finished if ``blocking=False``.

        Arguments:
            dst_dir (str): destination directory.
            blocking (bool, optional): dump embedding in blocking mode or not.
        """
        self.common_context.dump(dst_dir)
        if blocking:
            self.wait_for_dump_embedding()

    def load_embedding(self, src_dir: str, blocking: bool = True):
        """Load embeddings from ``src_dir``.
        By default, this function is synchronous and will wait for the completion
        of embedding loading before returning. This is done internally through
        a call to :meth:`.wait_for_load_embedding`.
        Set ``blocking=False`` to allow asyncronous computation,
        in which case the function will return immediately.

        Arguments:
            src_dir (str): directory to load embeddings.
            blocking (bool, optional): dump embedding in blocking mode or not.
        """
        self.common_context.load(src_dir)
        if blocking:
            self.wait_for_load_embedding()

    def dump_torch_state_dict(
        self,
        torch_instance: Union[torch.nn.Module, torch.optim.Optimizer],
        dst_dir: str,
        file_name: str,
        is_jit: bool = False,
    ):
        """Dump a Pytorch model or optimizer's state dict to the destination directory.

        Arguments:
            torch_instance (torch.nn.Module or torch.optim.Optimizer): dense model or
                optimizer to be dumped.
            dst_dir (str): destination directory.
            file_name (str): destination filename.
            is_jit (bool, optional): whether to dump model as jit script.
        """

        buffer = io.BytesIO()
        if not is_jit:
            torch.save(torch_instance.state_dict(), buffer)
        else:
            assert isinstance(
                torch_instance, torch.nn.Module
            ), f"dump_torch_object only support torch.nn.Moudle, but go obj type {type(torch_instance)}"
            jit_model = torch.jit.script(torch_instance)
            torch.jit.save(jit_model, buffer)
        bytes_model = buffer.getvalue()
        self.common_context.dump_to_file(bytes_model, dst_dir, file_name)

    def load_torch_state_dict(
        self,
        torch_instance: Union[torch.nn.Module, torch.optim.Optimizer],
        src_dir: str,
        map_location: Optional[str] = None,
    ):
        """Load a Pytorch state dict from the source directory and apply to `torch_instance`.

        Arguments:
            torch_instance (torch.nn.Module or torch.optim.Optimizer): dense model or
                optimizer to restore.
            src_dir (str): directory to load torch state dict.
            map_location (str, optional): load the dense checkpoint to specific device.
        """
        dense_bytes = self.common_context.read_from_file(src_dir)
        buffer = io.BytesIO(dense_bytes)
        buffer.seek(0)
        state_dict = torch.load(buffer, map_location=map_location)
        torch_instance.load_state_dict(state_dict)

    def wait_for_dump_embedding(self):
        """Wait for the embedding dump process."""
        self.common_context.wait_for_emb_dumping()

    def wait_for_load_embedding(self):
        """Wait for the embedding load process."""
        self.common_context.wait_for_emb_loading()

    def get_embedding_size(self) -> List[int]:
        """Get number of ids on all embedding servers."""
        return self.common_context.get_embedding_size()

    def clear_embeddings(self):
        """Clear all embeddings on all embedding servers."""
        self.common_context.clear_embeddings()

    def get_embedding_from_data(
        self, persia_batch: PersiaBatch, device_id: Optional[int] = None
    ) -> PersiaTrainingBatch:
        """Get embeddings of the serialized input batch data.

        Arguments:
            persia_batch (PersiaBatch): input data without embeddings..
            device_id (int, optional): the CUDA device to use for this process.

        Returns:
            PersiaTrainingBatch that contains id_type_feature_embeddings.
        """

        return self.common_context.get_embedding_from_data(
            persia_batch.data, device_id or self.device_id
        )

    def get_embedding_from_bytes(
        self, data: bytes, device_id: Optional[int] = None
    ) -> PersiaTrainingBatch:
        """Get embeddings of the serialized input batch data.

        Arguments:
            data (PersiaBatch): serialized input data without embeddings.
            device_id (int, optional): the CUDA device to use for this process.

        Returns:
            PersiaTrainingBatch that contains id_type_feature_embeddings.
        """

        return self.common_context.get_embedding_from_bytes(
            data, device_id or self.device_id
        )


class TrainCtx(EmbeddingCtx):
    r"""Subclass of :class:`EmbeddingCtx` that implements a `backward` function to update the
    embeddings.

    Example for :class:`TrainCtx`:

    .. code-block:: python

        import torch
        import persia
        from persia.data import DataLoder, StreamingDataset

        device_id = 0
        model = get_dnn_model()
        model.cuda(device_id)

        embedding_optimizer = persia.embedding.optim.SGD(lr=1e-3)
        dense_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCELoss(reduction="mean")

        prefetch_size = 15
        stream_dataset = StreamingDataset(prefetch_size)

        with TrainCtx(
            embedding_optimizer,
            dense_optimizer,
            model=model,
            device_id=device_id
        ) as ctx:
            dataloader = DataLoder(stream_dataset)
            for persia_training_batch in datalaoder:
                output, labels = ctx.forward(persia_training_batch)
                loss = loss_fn(output, labels[0])
                scaled_loss = ctx.backward(loss)

    If you want to train the PERSIA task in a distributed environment, you can
    set `distributed_option` to the corresponding option you want to use.
    Currently support Pytorch DDP (distributed data-parallel) (:class:`DDPOption`)
    and `Bagua <https://github.com/BaguaSys/bagua>`_ (:class:`BaguaDistributedOption`). The default is Pytorch DDP.
    The default configuration is determined by :func:`.get_default_distributed_option`
    when the environment ``WORLD_SIZE > 1``.

    You can configure the :class:`DDPOption` to your specific requirements.

    .. code-block::

        import persia
        from persia.distributed import DDPOption

        backend = "nccl"
        # backend = "gloo" # If you want to train the PERSIA on the CPU cluster.

        ddp_option = DDPOption(
            backend=backend,
            init_method="tcp"
        )

        with TrainCtx(
            embedding_optimizer,
            dense_optimizer,
            model=model,
            distributed_option=ddp_option
        ) as ctx:
            ...

    We also integrated Bagua to PERSIA as an alternative to PytorchDDP.
    `Bagua <https://github.com/BaguaSys/bagua>`_ is an advanced data-parallel framework,
    also developed by AI Platform @ Kuaishou.
    Using :class:`BaguaDistributedOption` in place of
    :class:`DDPOption` can significantly speed up the training (See
    `Bagua Benchmark <https://tutorials.baguasys.com/benchmark/>`_).
    For more details on the algorithms used by and available options of
    :class:`BaguaDistributedOption`, please refer to
    `Bagua tutorials <https://tutorials.baguasys.com/algorithms/>`_.

    Example for :class:`BaguaDistributedOption`:

    .. code-block::

        from persia.distributed import BaguaDistributedOption

        algorithm = "gradient_allreduce"
        bagua_args = {}
        bagua_option = BaguaDistributedOption(
            algorithm,
            **bagua_args
        )

        with TrainCtx(
            embedding_optimizer,
            dense_optimizer,
            model=model,
            distributed_option=bagua_option
        ) as ctx:
            ...

    """

    def __init__(
        self,
        embedding_optimizer: Optimizer,
        dense_optimizer: torch.optim.Optimizer,
        grad_scalar_update_factor: float = 4,
        backward_buffer_size: int = 10,
        backward_workers_size: int = 8,
        grad_update_buffer_size: int = 60,
        lookup_emb_directly: bool = True,
        mixed_precision: bool = True,
        distributed_option: Optional[DistributedBaseOption] = None,
        *args,
        **kwargs,
    ):
        """
        Arguments:
            embedding_optimizer (persia.embedding.optim.Optimizer): optimizer for the
                embedding parameters.
            dense_optimizer (torch.optim.Optimizer): optimizer for dense parameters.
            grad_scalar_update_factor (float, optional): update factor of ``Gradscalar``
                to ensure that loss scale is finite if set ``mixed_precision=True``.
            backward_buffer_size (int, optional): maximum number of gradients
                queued in the buffer between two backward steps.
            backward_workers_size (int, optional): number of workers sending embedding gradients
                in parallel.
            grad_update_buffer_size (int, optional): the size of gradient buffers. The buffer will cache the
                gradient tensor until the embedding update is finished.
            lookup_emb_directly (bool, optional): lookup embedding directly without a separate data loader.
            mixed_precision (bool): whether to enable mixed_precision.
            distributed_option (DistributedBaseOption, optional): option for distributed training.
        """
        super(TrainCtx, self).__init__(PreprocessMode.TRAIN, *args, **kwargs)

        assert (
            embedding_optimizer is not None
        ), "EmbeddingOptimizer should not be none in train context"
        assert grad_scalar_update_factor > 0, "grad scalar should greater than zero"
        assert (
            self.model is not None
        ), "Model not found, please init context with pytorch model"

        assert grad_scalar_update_factor > 0, "grad scalar should greater than zero"

        world_size = env.get_world_size()
        assert world_size != -1, "WORLD_SIZE not set"
        rank_id = env.get_rank()
        assert rank_id != -1, "RANK not set"

        self.world_size = world_size
        self.rank_id = rank_id

        assert not mixed_precision or (
            mixed_precision and torch.cuda.is_available()
        ), "Mixed precision training only support on cuda device."
        self.mixed_precision = mixed_precision

        if self.mixed_precision:
            self.grad_scalar_update_factor = grad_scalar_update_factor
            self.grad_scaler = torch.cuda.amp.GradScaler()
            self.update_times = 0

        if world_size > 1:
            distributed_option = distributed_option or get_default_distributed_option(
                device_id=self.device_id
            )
            not_env_file = not distributed_option.init_with_env_file()
            not_exists_master_addr = not distributed_option.master_addr
            if not_env_file and not_exists_master_addr:
                master_addr = self._get_master_addr()
            else:
                master_addr = None

            model, dense_optimizer = distributed_option.convert2distributed_model(
                self.model,
                world_size,
                rank_id,
                self.device_id,
                master_addr=master_addr,
                optimizer=dense_optimizer,
            )
            self.model = model
            _logger.info("Distributed training context init done.")
        else:
            _logger.info("SingleMachine training context init done.")

        self.dense_optimizer = dense_optimizer
        self.embedding_optimizer = embedding_optimizer

        self.wait_servers_ready()

        if lookup_emb_directly:
            init_rpc_client_num = self._init_embedding_worker_rpc_client()
            _logger.info(f"Successfully init {init_rpc_client_num} rpc client")

        self.backward_workers_size = backward_workers_size

        from persia.prelude import Backward

        self.grad_queue = Queue(grad_update_buffer_size)
        self.backward_engine = Backward(backward_buffer_size)

    def _enter(self):
        super()._enter()

        self.embedding_optimizer.apply()
        self.backward_engine.launch(self.backward_workers_size)

    def _exit(self):
        super()._exit()

        self.backward_engine.shutdown()

    def _get_master_addr(self) -> str:
        """Get leader(rank 0) ip address."""
        if self.rank_id == 0:
            master_addr = socket.gethostbyname(socket.gethostname())
            self.common_context.init_master_discovery_service(master_addr)
            _logger.info(f"init addr is {master_addr}")
        else:
            self.common_context.init_master_discovery_service()
            master_addr = self.common_context.master_addr
            _logger.info(f"master addr is {master_addr}")
        return master_addr

    def _init_embedding_worker_rpc_client(self) -> int:
        """Initialize the embedding worker rpc clients."""
        embedding_worker_addr_list = (
            self.common_context.get_embedding_worker_addr_list()
        )
        assert len(embedding_worker_addr_list) > 0, "Not available embedding worker."
        for embedding_worker_addr in embedding_worker_addr_list:
            self.common_context.init_rpc_client_with_addr(embedding_worker_addr)
        return len(embedding_worker_addr_list)

    def wait_servers_ready(self):
        """Wait until embedding servers are ready to serve."""

        self.common_context.init_nats_publisher(self.world_size)
        self.common_context.wait_servers_ready()

    def backward(
        self, loss: torch.Tensor, embedding_gradient_check_frequency: int = 20
    ) -> torch.Tensor:
        """Update the parameters of the current dense model and embedding model.

        Arguments:
            loss (torch.Tensor): loss of current batch.
            embedding_gradient_check_frequency (int, optional): how many batch_size to check
                gradient finite or not for current embedding.
        """
        if self.mixed_precision:
            loss = self.grad_scaler.scale(loss)
            scale = self.grad_scaler.get_scale()
        else:
            scale = 1  # Always equal to 1 when disable mixed_precision training

        loss.backward()

        finite = self._on_backward(scale, embedding_gradient_check_frequency)

        if self.mixed_precision:
            self.grad_scaler.step(self.dense_optimizer)

            if finite:
                self.grad_scaler.update()
            else:
                self.grad_scaler.update(scale / self.grad_scalar_update_factor)
        else:
            self.dense_optimizer.step()

        self.dense_optimizer.zero_grad()
        return loss

    def _on_backward(self, loss_scale: float, embedding_gradient_check_frequency: int):
        """Update the embeddings' gradients

        Arguments:
            loss_scale (float): the loss that scaled by GradScalar, loss_scale always equal
                to 1 for cpu training scenes.
            embedding_gradient_check_frequency (int): the frequency to check gradient finite
                or not for current embedding.
        """
        if self.grad_queue.full():
            self.grad_queue.get()

        finite = True

        if (
            self.mixed_precision
            and self.update_times % embedding_gradient_check_frequency == 0
        ):
            finite = _check_finite(
                [
                    embedding[-1].grad
                    for embedding in self.current_batch.id_type_feature_embedding_cache_torch_tensors
                ]
            )
            self.update_times += 1

        grad_slots = []  # cache grad slots
        empty_grads = []  # counting empty grads

        gradient_batch = self.current_batch.create_gradient_batch()

        for (
            emb_name,
            distinct_id_tensor,
            index,
            non_zero_index,
            emb_tensor,
        ) in self.current_batch.id_type_feature_embedding_cache_torch_tensors:
            if emb_tensor.grad is None:
                gradient_batch.add_skipped_gradient(emb_name)
                empty_grads.append(emb_name)
            else:
                if distinct_id_tensor is not None:
                    if distinct_id_tensor.shape[0] > 1:
                        grad = torch.zeros_like(distinct_id_tensor, dtype=torch.float32)
                        non_zero_grad = emb_tensor.grad.index_select(
                            0, non_zero_index.view(-1)
                        ).float()
                        non_zero_index = index.view(-1)[non_zero_index.view(-1)]
                        grad.index_add_(0, non_zero_index, non_zero_grad)
                        grad = grad[1:, :]
                        is_f16_gradient = False
                    else:
                        grad = None
                else:
                    grad = emb_tensor.grad  # type: torch.Tensor
                    is_f16_gradient = True

                if grad is not None:

                    grad_slots.append(grad)
                    gradient_batch.add_gradient(
                        emb_name,
                        grad.data_ptr(),
                        grad.shape,
                        is_f16_gradient,
                        loss_scale,
                    )

        if self.device_id is not None:
            torch.cuda.synchronize()

        self.backward_engine.update_id_type_feature_gradient_batched(gradient_batch)
        self.grad_queue.put(grad_slots)

        if len(empty_grads) > 0:
            _logger.warning(
                f"Current batch exists empty gradient tensors, num: {len(empty_grads)}, {empty_grads}"
            )
        return finite

    def dump_checkpoint(
        self,
        dst_dir: str,
        dense_model_filename: str = "dense.pt",
        jit_dense_model_filename: str = "jit_dense.pt",
        opt_filename: str = "opt.pt",
        blocking: bool = True,
        with_jit_model: bool = False,
    ):
        """Dump the dense and embedding checkpoint to destination directory.

        Arguments:
            dst_dir (str): destination directory.
            dense_model_filename (str, optional): dense model checkpoint filename.
            jit_dense_model_filename (str, optional): dense checkpoint filename after PyTorch jit.
            opt_filename (str, optional): optimizer checkpoint filename.
            blocking (bool, optional): dump embedding checkpoint in blocking mode or not.
            with_jit_model (bool, optional): dump dense checkpoint as jit script or not.
        """
        super().dump_checkpoint(
            dst_dir,
            dense_filename=dense_model_filename,
            jit_dense_filename=jit_dense_model_filename,
            blocking=blocking,
            with_jit_model=with_jit_model,
        )

        self.dump_torch_state_dict(self.dense_optimizer, dst_dir, opt_filename)

    def load_checkpoint(
        self,
        src_dir: str,
        map_location: Optional[str] = None,
        dense_model_filename: str = "dense.pt",
        opt_filename: str = "opt.pt",
        blocking: bool = True,
    ):
        """Load the dense and embedding checkpoint from source directory.

        Arguments:
            src_dir (str): source directory.
            map_location (str, optional): load the dense checkpoint to specific device.
            dense_model_filename (str, optional): dense checkpoint filename.
            opt_filename (str, optional): optimizer checkpoint filename.
            blocking (bool, optional): dump embedding checkpoint in blocking mode or not.
        """
        super().load_checkpoint(
            src_dir,
            map_location=map_location,
            dense_filename=dense_model_filename,
            blocking=blocking,
        )

        optimizer_filepath = os.path.join(src_dir, opt_filename)
        if os.path.exists(optimizer_filepath):
            self.load_torch_state_dict(
                self.dense_optimizer, optimizer_filepath, map_location=map_location
            )


def cnt_ctx() -> Optional[BaseCtx]:
    """Get the :class:`BaseCtx` recently entered."""
    return _CURRENT_CXT


def eval_ctx(*args, **kwargs) -> EmbeddingCtx:
    """Get the :class:`EmbeddingCtx` with the :attr:`.EVAL` mode."""
    return EmbeddingCtx(PreprocessMode.EVAL, *args, **kwargs)


class InferCtx(EmbeddingCtx):
    r"""Subclass of :class:`EmbeddingCtx` that provides the inference functionality without `nats-servers`.

    Example for :class:`InferCtx`:

    .. code-block:: python

        import numpy as np
        from persia.ctx import InferCtx
        from persia.embedding.data import PersiaBatch, IDTypeFeatureWithSingleID

        device_id = 0
        id_type_feature = IDTypeFeatureWithSingleID(
            "id_type_feature",
            np.array([1, 2, 3], np.uint64)
        )
        persia_batch = PersiaBatch([id_type_feature], requires_grad=False)

        embedding_worker_address_list = [
            "localhost: 8888",
            "localhost: 8889",
            "localhost: 8890"
        ]
        with InferCtx(embedding_worker_address_list, device_id=device_id) as infer_ctx:
            persia_training_batch = persia_context.get_embedding_from_bytes(
                persia_batch.to_bytes(),
            )
            (
                non_id_type_feature_tensors,
                id_type_feature_embedding_tensors,
                label_tensors
            )= persia_context.prepare_features(batch)

    .. note::
        The example cannot be run directly, you should launch the `embedding-worker` and
        `embedding-parameter-server` to ensure the example gets correct result.
    """

    def __init__(
        self,
        embedding_worker_address_list: List[str],
        *args,
        **kwargs,
    ):
        """
        Arguments:
            embedding_worker_addrs (List[str]): embedding worker address(ip:port) list.
        """
        super(InferCtx, self).__init__(PreprocessMode.INFERENCE, *args, **kwargs)

        for addr in embedding_worker_address_list:
            self.common_context.init_rpc_client_with_addr(addr)

    r"""Wait for embedding worker and embedding server ready for serving."""

    def wait_for_serving(self):
        self.common_context.wait_for_serving()
