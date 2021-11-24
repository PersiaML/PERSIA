import os
import io
import socket

from enum import Enum
from queue import Queue
from typing import List, Tuple, Optional, Union

import torch

import persia.env as env

from persia.logger import get_default_logger
from persia.embedding.optim import Optimizer
from persia.embedding import EmbeddingConfig, get_default_embedding_config
from persia.prelude import (
    PersiaCommonContext,
    PersiaBatch,
    PersiaTrainingBatch,
    Tensor,
)
from persia.distributed import DistributedBaseOption, get_default_distributed_option

_CURRENT_CXT = None

_logger = get_default_logger()


def _check_finite(tensors: List[torch.Tensor]) -> bool:
    """Check if all tensors in the input list contain only finite elements.

    Arguments:
        tensors (List[torch.Tensor]): List of tensor to be checked.

    Returns:
        bool: ``True`` if all elements in ``tensors`` are finite or None.
    """
    return all([torch.isfinite(t).all() if t is not None else True for t in tensors])


def _cast_dlpack2torch_tensor(
    tensor: Tensor, requires_grad: bool = False
) -> torch.Tensor:
    """Convert the DLPack PythonCapsule to torch tensor

    Arguments:
        Tensor (Tensor): Tensor wrapper that contains dlpack information.
        requires_grad (bool, optional): Whether current tensor requires grad or not.
    Returns: pytorch tensor
    """

    import torch.utils.dlpack as dlpack

    tensor = dlpack.from_dlpack(tensor.dlpack)
    tensor.requires_grad = requires_grad
    return tensor


class PreprocessMode(Enum):
    r"""Mode of preprocessing.

    Used by ``EmbeddingCtx.prepare_features`` to generate features of different datatypes.

    When set to ``TRAIN``, ``prepare_features`` will return a torch tensor with ``requires_grad`` attribute set to ``True``.
    When set to ``EVAL``, ``prepare_features`` will return a torch tensor with ``requires_grad`` attribute set to ``False``.
    ``INFERENCE`` behaves almost identical to ``PreprocessMode.EVAL``, except that ``INFERENCE`` allows ""EmbeddingCtx`` to process the ``PythonTrainBatch`` without a target tensor.
    """
    TRAIN = 1
    EVAL = 2
    INFERENCE = 3


class BaseCtx:
    r"""Initializes a common context for other persia context, e.g. `DataCtx`, `EmbeddingCtx` and `TrainCtx`.
    This class should not be instantiated directly.
    """

    def __init__(
        self, threadpool_worker_size: int = 10, device_id: Optional[int] = None
    ):
        """
        Arguments:
            threadpool_worker_size (int): Rpc threadpool worker size.
            device_id (int, optional): The CUDA device to use for this process.
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
        if env.get_rank() is not None and env.get_rank() >= 0:
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
    r"""Provides the communicate ability for data generator component to send the PersiaBatchData
    to the nn worker and embedding worker.

    Example:
        >>> from persia.prelude import PersiaBatch
        >>> loader = make_simple_loader()
        >>> with DataCtx() as ctx:
        >>>     for (non_id_type_features, id_type_features, label) in loader:
        >>>         batch_data = PersiaBatch()
        >>>         batch_data.add_non_id_type_features(non_id_type_features)
        >>>         batch_data.add_id_type_features(id_type_features)
        >>>         batch_data.add_label(label)
        >>>         ctx.send_data(batch_data)
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(DataCtx, self).__init__(*args, **kwargs)
        self.prepare()
        _logger.info("Data ctx prepare done.")

    def prepare(self):
        """Do some preparation to init `DataCtx`."""

        self.common_context.init_nats_publisher(None)
        self.common_context.wait_servers_ready()

    def send_id_type_features_to_embedding_worker(self, data: PersiaBatch):
        """Send PersiaBatch from data loader to embedding worker side.

        Arguments:
            data (PersiaBatch): PersiaBatch that haven't been process.
        """
        self.common_context.send_id_type_features_to_embedding_worker(data)

    def send_non_id_type_features_to_nn_worker(self, data: PersiaBatch):
        """Send `PersiaBatch` from data loader to nn worker side.

        Arguments:
            data (PersiaBatch): PersiaBatch that have been sent to embedding worker.
        """
        self.common_context.send_non_id_type_features_to_nn_worker(data)

    def send_data(self, data: PersiaBatch):
        """Send PersiaBatch from data loader to nn worker and embedding worker side.

        Arguments:
            data (PersiaBatch): PersiaBatch that haven't been process.
        """
        self.send_id_type_features_to_embedding_worker(data)
        self.send_non_id_type_features_to_nn_worker(data)


class EmbeddingCtx(BaseCtx):
    r"""Provides the embedding-related functionality. EmbeddingCtx can run offline test or online inference
    depending on different preprocess_mode. The simplest way to get this context is by using ``persia.ctx.eval_ctx()`` or
    ``persia.ctx.inference_ctx`` to get the ``EmbeddingCtx`` instance.

    Example:
        >>> from persia.prelude import PyPersiaBatchData
        >>> model = get_dnn_model()
        >>> loader = make_dataloader()
        >>> embedding_config = EmbeddingConfig()
        >>> with EmbeddingCtx(
        ...     model=model,
        ...     PreprocessMode.EVAL,
        ...     embedding_config
        ... ) as ctx:
        >>>     for (non_id_type_feature, id_type_features, label) in loader:
        >>>         batch_data = PyPersiaBatchData()
        >>>         batch_data.add_non_id_type_feature(non_id_type_feature)
        >>>         batch_data.add_id_type_features(id_type_features)
        >>>         batch_data.add_label(label)
        >>>         python_training_batch = ctx.get_embedding_from_data(batch_data)
        >>>         (output, label) = ctx.forward(python_training_batch)
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
            preprocess_mode (PreprocessMode): Different preprocess mode effect the behavior of ``prepare_features``.
            model (torch.nn.Module): Torch model matched with embeddings in this context.
            embedding_config (EmbeddingConfig, optional): The embedding configuration that will be sent to the embedding server.
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
        """Apply Embedding config to embedding servers.
        Arguments:
            embedding_config (EmbeddingConfig): The embedding configuration that will be sent to the embedding server.
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
        """Call `prepare_features` and then do a forward step of the model in context.

        Arguments:
            batch (PythonTrainBatch): Training data provided by PersiaML upstream including
                dense, target, sparse data and meta info.

        Returns:
            the tuple of output data and target data.
        """
        assert self.model is not None, "model not found, please init context with model"
        non_id_type_tensors, embedding_tensors, labels = self.prepare_features(batch)
        output = self.model(non_id_type_tensors, embedding_tensors)
        return (output, labels)

    def prepare_features(
        self, batch: PersiaTrainingBatch
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
        """Converts the non_id_type_features, sparse and target raw data in``PersiaTrainingBatch`` to `torch.Tensor``.

        TODO: Add the feature conversion detail
        For example

        NotIdTypeFeatures -> ...
        IDTypeFeatures -> ...
        Labels -> ...

        Arguments:
            batch (PersiaTrainingBatch): Training data provided by PersiaML upstream including
                dense, target, sparse data and meta info.

        Returns:
            the tuple of dense data, list of sparse data and target data.
        """
        if self.preprocess_mode == PreprocessMode.INFERENCE:
            batch.label_tensors = None
        else:
            # pytype: disable=attribute-error
            batch.labels = batch.consume_all_labels()
            # pytype: enable=attribute-error
            batch.label_tensors = [
                _cast_dlpack2torch_tensor(label) for label in batch.labels
            ]

        is_training = self.preprocess_mode == PreprocessMode.TRAIN  # cache property

        # pytype: disable=attribute-error
        batch.non_id_type_features = batch.consume_all_non_id_type_features()
        # pytype: enable=attribute-error
        batch.non_id_type_tensors = [
            _cast_dlpack2torch_tensor(non_id_type_feature)
            for non_id_type_feature in batch.non_id_type_features
        ]

        # pytype: disable=attribute-error
        batch.id_type_features = batch.consume_all_id_type_features()
        # pytype: enable=attribute-error

        batch.emb_slots = []  # cache embedding to prevent tensor expired
        emb_tensors = []  # cache origin embedding for later backward procedure
        id_type_tensors = []  # id type tensos for later forward procedure

        for id_type_feature in batch.id_type_features:
            if id_type_feature.is_raw_embedding():
                # no duplicate id in raw_id_tensor
                (
                    raw_embedding,
                    index,
                    non_empty_index,
                    sample_id_num,
                ) = id_type_feature.get_raw_embedding()

                batch.emb_slots.append([raw_embedding, index, non_empty_index])
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
                emb_tensors.append(
                    (
                        raw_embedding.name,
                        distinct_id_tensor,
                        index_tensor,
                        non_empty_index_tensor,
                        index_select_raw_tensor,
                    )
                )
                id_type_tensors.append(raw_fixed_size_tensor_with_mask)
            else:
                emb = id_type_feature.get_sum_embedding()
                batch.emb_slots.append([emb])
                attention_sum_tensor = _cast_dlpack2torch_tensor(
                    emb, requires_grad=is_training
                )
                id_type_tensors.append(attention_sum_tensor)
                emb_tensors.append((emb.name, None, None, None, attention_sum_tensor))

        batch.id_type_tensors = id_type_tensors
        batch.emb_tensors = emb_tensors
        self.current_batch = batch

        return batch.non_id_type_tensors, batch.id_type_tensors, batch.label_tensors

    def dump_checkpoint(
        self,
        dst_dir: str,
        dense_filename: str = "dense.pt",
        jit_dense_filename: str = "jit_dense.pt",
        blocking: bool = True,
        with_jit_model: bool = False,
    ):
        """Dump the dense and sparse checkpoint to destination directory.

        Arguments:
            dst_dir (str): Destination directory.
            dense_filename (str, optional): Dense checkpoint filename.
            jit_dense_filename (str, optional): Jit dense checkpoint filename.
            blocking (bool, optional): Dump embedding checkpoint in blocking mode or not.
            with_jit_model (bool, optional): Dump jit script dense checkpoint or not.
        """
        assert self.model is not None, "model not found, please init context with model"

        if with_jit_model:
            self.dump_dense(self.model, dst_dir, jit_dense_filename, True)
        self.dump_dense(self.model, dst_dir, dense_filename)

        self.dump_embedding(dst_dir, blocking=blocking)

    def load_checkpoint(
        self,
        src_dir: str,
        map_location: Optional[str] = None,
        dense_filename: str = "dense.pt",
        blocking: bool = True,
    ):
        """Load the dense and sparse checkpoint from source directory.

        Arguments:
            src_dir (str): Source directory.
            map_location (str, optional): Load the dense checkpoint to specific device.
            dense_filename (str, optional): Dense checkpoint filename.
            blocking (bool, optional): Dump embedding checkpoint in blocking mode or not.
        """
        assert self.model is not None, "model not found, please init context with model"

        dense_model_filepath = os.path.join(src_dir, dense_filename)
        if os.path.exists(dense_model_filepath):
            self.load_dense(self.model, dense_model_filepath, map_location=map_location)

        self.load_embedding(src_dir, blocking=blocking)

    def dump_embedding(self, dst_dir: str, blocking: bool = True):
        """Dump embeddings to ``dst_dir``. Use ``TrainCtx.wait_for_dump_embedding`` to wait until finished
        if ``blocking=False``.

        Arguments:
            dst_dir (str): Destination directory.
            blocking (bool, optional): Dump embedding in blocking mode or not.
        """
        self.common_context.dump(dst_dir)
        if blocking:
            self.wait_for_dump_embedding()

    def load_embedding(self, src_dir: str, blocking: bool = True):
        """Load embeddings from ``src_dir``. Use ``TrainCtx.wait_for_load_embedding`` to wait until finished
        if ``blocking=False``.

        Arguments:
            src_dir (str): Directory to load embeddings.
            blocking (bool, optional): Dump embedding in blocking mode or not.
        """
        self.common_context.load(src_dir)
        if blocking:
            self.wait_for_load_embedding()

    def dump_dense(
        self,
        dense: Union[torch.nn.Module, torch.optim.Optimizer],
        dst_dir: str,
        file_name: str,
        is_jit: bool = False,
    ):
        """Dump torch model or optimizer to ``dst_dir`` as ``file_name``.

        Arguments:
            dense (torch.nn.Module or torch.optim.Optimizer): dense model or optimizer to be dumped.
            dst_dir (str): Destination directory.
            file_name (str): Destination filename.
            is_jit (bool, optional): whether to dump model as jit script.
        """
        buffer = io.BytesIO()
        if not is_jit:
            torch.save(dense.state_dict(), buffer)
        else:
            assert isinstance(
                dense, torch.nn.Module
            ), "saving an optimizer as jit script"
            jit_model = torch.jit.script(dense)
            torch.jit.save(jit_model, buffer)
        bytes_model = buffer.getvalue()
        self.common_context.dump_to_file(bytes_model, dst_dir, file_name)

    def load_dense(
        self,
        dense: Union[torch.nn.Module, torch.optim.Optimizer],
        src_filepath: str,
        map_location: Optional[str] = None,
    ):
        """Load the torch state dict from source file path.

        Arguments:
            dense (torch.nn.Module or torch.optim.Optimizer): dense model or optimizer to restore.
            src_filepath (str): Source file path.
            map_location (str, optional): Load the dense checkpoint to specific device.
        """
        dense_bytes = self.common_context.read_from_file(src_filepath)
        buffer = io.BytesIO(dense_bytes)
        buffer.seek(0)
        state_dict = torch.load(buffer, map_location=map_location)
        dense.load_state_dict(state_dict)

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
        self, data: PersiaBatch, device_id: Optional[int] = None
    ) -> PersiaTrainingBatch:
        """Get embeddings of the input batch data.

        Arguments:
            data (PyPersiaBatchData): Input data without embeddings.
            device_id (int, optional): The CUDA device to use for this process.

        Returns:
            Input data with embeddings.
        """
        return self.common_context.get_embedding_from_data(data, device_id)

    def get_embedding_from_bytes(
        self, data: bytes, device_id: Optional[int] = None
    ) -> PersiaTrainingBatch:
        """Get embeddings of the serialized input batch data.

        Arguments:
            data (PyPersiaBatchData): Serialized input data without embeddings.
            device_id (int, optional): The CUDA device to use for this process.

        Returns:
            Input data with embeddings.
        """
        return self.common_context.get_embedding_from_bytes(data, device_id)


class TrainCtx(EmbeddingCtx):
    r"""Subclass of ``EmbeddingCtx`` that provide the backward ability to update the sparse embedding.

    Example:
        >>> import torch
        >>> model = get_dnn_model()
        >>> embedding_optimizer = persia.embedding.optim.SGD(lr=1e-3)
        >>> dense_optimizer = torch.optim.SGD(lr=1e-3)
        >>> loss_fn = torch.nn.BCELoss(reduction="mean")
        >>> with TrainCtx(
        >>>     embedding_optimizer,
        >>>     dense_optimizer,
        >>> ) as ctx:
        >>>     parallel_model = ctx.model
        >>>     for batch_data in datalaoder:
        >>>         dense, sparse, target = ctx.prepare_features(data)
        >>>         output = parallel_model(dense, sparse)
        >>>         loss = loss_fn(output, target)
        >>>         scaled_loss = ctx.backward(loss)
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
            embedding_optimizer (persia.embedding.optim.Optimizer): Optimizer for the embeddings.
            dense_optimizer (torch.optim.Optimizer): Optimizer for dense parameters.
            grad_scalar_update_factor (float, optional): Update factor of ``Gradscalar`` to ensure loss scale finitely if set ``mixed_precision=True``.
            backward_buffer_size (int, optional): Max number of not updated gradients queued.
            backward_workers_size (int, optional): Number of workers sending embedding gradients in parallel.
            grad_update_buffer_size (int, optional): Number of reference cache , hold the gradient tensor reference to avoid
                meet dangle data in gradient backward phase.
            lookup_emb_directly (bool, optional): Lookup embedding directly without isolation data loader.
            mixed_precision (bool): Enable mixed_precision or not.
            distributed_option (DistributedBaseOption, optional): DistributedOption to converted model to dataparallel model.
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
            distributed_option = distributed_option or get_default_distributed_option()
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
            self.common_context.init_master_discovery_service(None)
            master_addr = self.common_context.master_addr
            _logger.info(f"master addr is {master_addr}")
        return master_addr

    def _init_embedding_worker_rpc_client(self) -> int:
        embedding_worker_addr_list = (
            self.common_context.get_embedding_worker_addr_list()
        )
        assert len(embedding_worker_addr_list) > 0, "Not available embedding worker."
        for embedding_worker_addr in embedding_worker_addr_list:
            self.common_context.init_rpc_client_with_addr(embedding_worker_addr)
        return len(embedding_worker_addr_list)

    def wait_servers_ready(self):
        """query embedding server to check if servers are ready"""

        self.common_context.init_nats_publisher(self.world_size)
        self.common_context.wait_servers_ready()

    def backward(
        self, loss: torch.Tensor, embedding_gradient_check_frequency: int = 20
    ) -> torch.Tensor:
        """Compute the gradient of current dense and sparse tensors.

        This method supports mixed precision training. Depending on whether the current embedding gradient is finite or not, ``GradScalar``
        can update the scale automatically to restrict to parameters in finite range.

        Arguments:
            loss (torch.Tensor): Loss of current batch.
            embedding_gradient_check_frequency (int, optional): The frequency to check gradient finite or not for current embedding.
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
            loss_scale (float): The loss that scaled by GradScalar, loss_scale always equal to 1 for cpu training scenes.
            embedding_gradient_check_frequency (int): The frequency to check gradient finite or not for current embedding.
        """
        if self.grad_queue.full():
            self.grad_queue.get()

        finite = True

        if (
            self.mixed_precision
            and self.update_times % embedding_gradient_check_frequency == 0
        ):
            finite = _check_finite(
                [emb[-1].grad for emb in self.current_batch.emb_tensors]
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
        ) in self.current_batch.emb_tensors:
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
        dense_filename: str = "dense.pt",
        jit_dense_filename: str = "jit_dense.pt",
        opt_filename: str = "opt.pt",
        blocking: bool = True,
        with_jit_model: bool = False,
    ):
        """Dump the dense and sparse checkpoint to destination directory.

        Arguments:
            dst_dir (str): Destination directory.
            dense_filename (str, optional): Dense checkpoint filename.
            jit_dense_filename (str, optional): Jit dense checkpoint filename.
            opt_filename (str, optional): Optimizer checkpoint filename.
            blocking (bool, optional): Dump embedding checkpoint in blocking mode or not.
            with_jit_model (bool, optional): Dump dense checkpoint as jit script or not.
        """
        super().dump_checkpoint(
            dst_dir,
            dense_filename=dense_filename,
            jit_dense_filename=jit_dense_filename,
            blocking=blocking,
            with_jit_model=with_jit_model,
        )

        self.dump_dense(self.dense_optimizer, dst_dir, opt_filename)

    def load_checkpoint(
        self,
        src_dir: str,
        map_location: Optional[str] = None,
        dense_filename: str = "dense.pt",
        opt_filename: str = "opt.pt",
        blocking: bool = True,
    ):
        """Load the dense and sparse checkpoint from source directory.

        Arguments:
            src_dir (str): Source directory.
            map_location (str, optional): Load the dense checkpoint to specific device.
            dense_filename (str, optional): Dense checkpoint filename.
            opt_filename (str, optional): Optimizer checkpoint filename.
            blocking (bool, optional): Dump embedding checkpoint in blocking mode or not.
        """
        super().load_checkpoint(
            src_dir,
            map_location=map_location,
            dense_filename=dense_filename,
            blocking=blocking,
        )

        optimizer_filepath = os.path.join(src_dir, opt_filename)
        if os.path.exists(optimizer_filepath):
            self.load_dense(self.dense_optimizer, optimizer_filepath)


def cnt_ctx() -> Optional[BaseCtx]:
    """Get the BaseCtx recently entered."""
    return _CURRENT_CXT


def eval_ctx(*args, **kwargs) -> EmbeddingCtx:
    """Get the ``EmbeddingCtx`` with the ``PreprocessMode.EVAL`` mode."""
    return EmbeddingCtx(PreprocessMode.EVAL, *args, **kwargs)


class InferCtx(EmbeddingCtx):
    r"""Subclass of ``EmbeddingCtx`` that provide the forward ability without nats servers.

    Example:
        >>> from persia.ctx import InferCtx
        >>> persia_context = InferCtx()
        >>> batch = persia_context.get_embedding_from_bytes(batch, device_id)
        >>> model_input = persia_context.prepare_features(batch)
    """

    def __init__(
        self,
        embedding_worker_addrs: List[str],
        *args,
        **kwargs,
    ):
        """
        Arguments:
            embedding_worker_addrs (List[str]): embedding worker address(ip:port) list.
        """
        super(InferCtx, self).__init__(PreprocessMode.INFERENCE, *args, **kwargs)

        for addr in embedding_worker_addrs:
            self.common_context.init_rpc_client_with_addr(addr)

    r"""Wait for embedding worker and embedding server ready for serving."""

    def wait_for_serving(self):
        self.common_context.wait_for_serving()
