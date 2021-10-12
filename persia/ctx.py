import os
import io
import socket

from enum import Enum
from queue import Queue
from typing import List, Tuple, Optional, NewType, Union

import torch

from retrying import retry

import persia.env as env

from persia.logger import get_default_logger
from persia.sparse.optim import Optimizer
from persia.distributed import DistributedBaseOption, get_default_distributed_option
from persia.prelude import PyPersiaCommonContext, PyPersiaBatchData

_CURRENT_CXT = None

_logger = get_default_logger()

PythonTrainBatch = NewType("PythonTrainBatch", object)


def _check_finite(tensors: List[torch.Tensor]) -> bool:
    """Check if all tensors in the input list contain only finite elements.

    Arguments:
        tensors (List[torch.Tensor]): List of tensor to be checked.

    Returns:
        bool: ``True`` if all elements in ``tensors`` are finite or None.
    """
    return all([torch.isfinite(t).all() if t is not None else True for t in tensors])


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
        self,
        threadpool_worker_size: int = 10,
    ):
        """
        Arguments:
            threadpool_worker_size (int): Rpc threadpool worker size.
        """
        self.origin_context = None

        replica_index = (
            env.get_rank() if env.get_rank() != -1 else env.get_replica_index()
        )
        replica_size = (
            env.get_world_size()
            if env.get_world_size() != -1
            else env.get_replica_size()
        )

        self.common_context = PyPersiaCommonContext(
            threadpool_worker_size, replica_index, replica_size
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
    to the trainer and embedding middleware.

    Example:
        >>> from persia.prelude import PyPersiaBatchData
        >>> loader = make_simple_loader()
        >>> with DataCtx() as ctx:
        >>>     for (dense, batch_sparse_ids, target) in loader:
        >>>         batch_data = PyPersiaBatchData()
        >>>         batch_data.add_dense([dense])
        >>>         batch_data.add_sparse(batch_sparse_ids)
        >>>         batch_data.add_target(target)
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

    @retry(wait_fixed=2000)
    def prepare(self):
        """Do some preparation to init `DataCtx`."""

        self.common_context.init_nats_publisher(None)
        self.common_context.wait_servers_ready()

    @retry(wait_fixed=2000)
    def send_sparse_to_middleware(self, data: PyPersiaBatchData):
        """Send PersiaBatchData from data compose to middleware side.

        Arguments:
            data (PyPersiaBatchData): PersiaBatchData that haven't been process.
        """
        self.common_context.send_sparse_to_middleware(data)

    @retry(wait_fixed=2000)
    def send_dense_to_trainer(self, data: PyPersiaBatchData):
        """Send PersiaBatchData from data compose to trainer side.

        Arguments:
            data (PyPersiaBatchData): PersiaBatchData that have been sent to middleware.
        """
        self.common_context.send_dense_to_trainer(data)

    def send_data(self, data: PyPersiaBatchData):
        """Send PersiaBatchData from data compose to trainer and middleware side.

        Arguments:
            data (PyPersiaBatchData): PersiaBatchData that haven't been process.
        """
        self.send_sparse_to_middleware(data)
        self.send_dense_to_trainer(data)


class EmbeddingConfig:
    r"""Embedding hyperparameters, argument of ``EmbeddingCtx``."""

    def __init__(
        self,
        emb_initialization: Tuple[float, float] = (-0.01, 0.01),
        admit_probability: float = 1.0,
        weight_bound: float = 10,
    ):
        """
        Arguments:
            emb_initialization (Tuple[float, float], optional): Lower and upper bound of embedding uniform initialization.
            admit_probability (float, optional): The probability (0<=, <=1) of admitting a new embedding.
            weight_bound (float, optional): Restrict each element value of an embedding in [-weight_bound, weight_bound].
        """
        self.emb_initialization = emb_initialization
        self.admit_probability = admit_probability
        self.weight_bound = weight_bound


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
        >>>     for (dense, batch_sparse_ids, target) in loader:
        >>>         batch_data = PyPersiaBatchData()
        >>>         batch_data.add_dense([dense])
        >>>         batch_data.add_sparse(batch_sparse_ids)
        >>>         batch_data.add_target(target)
        >>>         python_train_batch = ctx.get_embedding_from_data(batch_data)
        >>>         (output, target) = ctx.forward(python_train_batch)
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
        self.embedding_config = embedding_config

        self.current_batch = None

    def _enter(self):
        if self.embedding_config is not None:
            self.configure_embedding_servers(self.embedding_config)

    @retry(wait_fixed=2000)
    def configure_embedding_servers(
        self,
        embedding_config: EmbeddingConfig,
    ):
        """Apply Embedding config to embedding servers.
        Arguments:
            embedding_config (EmbeddingConfig): The embedding configuration that will be sent to the embedding server.
        """
        self.common_context.configure_embedding_servers(
            embedding_config.emb_initialization[0],
            embedding_config.emb_initialization[1],
            embedding_config.admit_probability,
            embedding_config.weight_bound > 0,
            embedding_config.weight_bound,
        )

    def forward(
        self, batch: PythonTrainBatch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Call `prepare_features` and then do a forward step of the model in context.

        Arguments:
            batch (PythonTrainBatch): Training data provided by PersiaML upstream including
                dense, target, sparse data and meta info.

        Returns:
            the tuple of output data and target data.
        """
        assert self.model is not None, "model not found, please init context with model"
        dense, sparse, target = self.prepare_features(batch)
        output = self.model(dense, sparse)
        return (output, target)

    def prepare_features(
        self, batch: PythonTrainBatch
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
        """Converts the dense, sparse and target raw data in``PythonTrainBatch`` to `torch.Tensor``.

        Arguments:
            batch (PythonTrainBatch): Training data provided by PersiaML upstream including
                dense, target, sparse data and meta info.

        Returns:
            the tuple of dense data, list of sparse data and target data.
        """
        import persia_torch_ext as pte  # pytype: disable=import-error

        if self.preprocess_mode == PreprocessMode.INFERENCE:
            batch.target_tensor = None
        else:
            # pytype: disable=attribute-error
            batch.target = batch.consume_all_targets()
            # pytype: enable=attribute-error
            assert len(batch.target) == 1
            batch.target = batch.target[0]

            batch.target_tensor = pte.ptr_to_tensor_f32(
                batch.target.data_ptr(), batch.target.shape(), False
            )

        is_training = self.preprocess_mode == PreprocessMode.TRAIN  # cache property

        # pytype: disable=attribute-error
        batch.dense = batch.consume_all_dense_features()
        # pytype: enable=attribute-error
        batch.dense = batch.dense[0]
        batch.dense_tensor = pte.ptr_to_tensor_f32(
            batch.dense.data_ptr(), batch.dense.shape(), False
        )

        # pytype: disable=attribute-error
        batch.emb = batch.consume_all_sparse_features()
        # pytype: enable=attribute-error

        batch.emb_slot = []
        # sparse embedding processing
        emb_tensors, forward_tensors = [], []

        for emb in batch.emb:
            if emb.is_raw_embedding():
                # no duplicate id in raw_id_tensor
                (
                    raw_embedding,
                    index,
                    non_empty_index,
                    sample_id_num,
                ) = emb.get_raw_embedding()

                batch.emb_slot.append([raw_embedding, index, non_empty_index])

                distinct_id_tensor = pte.ptr_to_tensor_f16(
                    raw_embedding.data_ptr(), raw_embedding.shape(), False
                )
                index_tensor = pte.ptr_to_tensor_long(
                    index.data_ptr(),
                    index.shape(),
                )  # tensor shape (1, batch_size * sample_fixed_size)
                max_index = index_tensor.max()
                size_of_distinct_id_tensor = distinct_id_tensor.shape[0]
                torch.cuda.synchronize()

                assert (
                    max_index < size_of_distinct_id_tensor
                ), "raw embedding select index larger than tensor"
                non_empty_index_tensor = pte.ptr_to_tensor_long(
                    non_empty_index.data_ptr(), non_empty_index.shape()
                )  # tensor shape (-1), variable length

                batch_size = len(sample_id_num)
                dim = distinct_id_tensor.shape[-1]
                sample_fixed_size = index_tensor.shape[1] // batch_size
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
                        raw_embedding.name(),
                        distinct_id_tensor,
                        index_tensor,
                        non_empty_index_tensor,
                        index_select_raw_tensor,
                    )
                )
                forward_tensors.append(raw_fixed_size_tensor_with_mask)
            else:
                emb = emb.get_sum_embedding()
                batch.emb_slot.append([emb])

                sum_tensor = pte.ptr_to_tensor_f16(
                    emb.data_ptr(), emb.shape(), is_training
                )
                forward_tensors.append(sum_tensor)
                emb_tensors.append((emb.name(), None, None, None, sum_tensor))

        batch.forward_tensors = forward_tensors
        batch.emb_tensors = emb_tensors
        self.current_batch = batch

        return batch.dense_tensor, batch.forward_tensors, batch.target_tensor

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
            self.load_dense(self.model, dense_model_filepath)

        self.load_embedding(src_dir, blocking=blocking)

    def dump_embedding(self, dst_dir: str, blocking: bool = False):
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
    ):
        """Load the torch state dict from source file path.

        Arguments:
            dense (torch.nn.Module or torch.optim.Optimizer): dense model or optimizer to restore.
            src_filepath (str): Source file path.
        """
        dense_bytes = self.common_context.read_from_file(src_filepath)
        buffer = io.BytesIO(dense_bytes)
        buffer.seek(0)
        state_dict = torch.load(buffer)
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

    @retry(wait_fixed=2000)
    def get_embedding_from_data(
        self, data: PyPersiaBatchData, device_id: int = 0
    ) -> PythonTrainBatch:
        """Get embeddings of the input batch data.

        Arguments:
            data (PyPersiaBatchData): Input data without embeddings.
            device_id (int, optional): The CUDA device to use for this process.

        Returns:
            Input data with embeddings.
        """
        return self.common_context.get_embedding_from_data(data, device_id)

    @retry(wait_fixed=2000)
    def get_embedding_from_bytes(
        self, data: bytes, device_id: int = 0
    ) -> PythonTrainBatch:
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
        >>> sparse_optimizer = persia.sparse.optim.SGD(lr=1e-3)
        >>> dense_optimizer = torch.optim.SGD(lr=1e-3)
        >>> loss_fn = torch.nn.BCELoss(reduction="mean")
        >>> with TrainCtx(
        >>>     sparse_optimizer,
        >>>     dense_optimizer,
        >>> ) as ctx:
        >>>     parallel_model = ctx.model
        >>>     for batch_data in dataloder:
        >>>         dense, sparse, target = ctx.prepare_features(data)
        >>>         output = parallel_model(dense, sparse)
        >>>         loss = loss_fn(output, target)
        >>>         scaled_loss = ctx.backward(loss)
    """

    def __init__(
        self,
        sparse_optimizer: Optimizer,
        dense_optimizer: torch.optim.Optimizer,
        device_id: int = 0,
        grad_scalar_update_factor: float = 4,
        backward_buffer_size: int = 10,
        backward_workers_size: int = 8,
        grad_update_buffer_size: int = 60,
        lookup_emb_directly: bool = True,
        distributed_option: Optional[DistributedBaseOption] = None,
        *args,
        **kwargs,
    ):
        """
        Arguments:
            sparse_optimizer (persia.sparse.optim.Optimizer): Optimizer for the embeddings.
            dense_optimizer (torch.optim.Optimizer): Optimizer for dense parameters.
            device_id (int, optional): The CUDA device to use for training.
            grad_scalar_update_factor (float, optional): Update factor of ``Gradscalar`` to ensure loss scale finitely if set ``mixed_precision=True``.
            backward_buffer_size (int, optional): Max number of not updated gradients queued.
            backward_workers_size (int, optional): Number of workers sending embedding gradients in parallel.
            grad_update_buffer_size (int, optional): Number of reference cache , hold the gradient tensor reference to avoid
                meet dangle data in gradient backward phase.
            lookup_emb_directly (bool, optional): Lookup embedding directly without isolation data compose.
            distributed_option (DistributedBaseOption, optional): DistributedOption to converted model to dataparallel model.
        """
        super(TrainCtx, self).__init__(PreprocessMode.TRAIN, *args, **kwargs)

        assert (
            sparse_optimizer is not None
        ), "Sparse_optimizer should not be none in train context"
        assert (
            0 <= device_id < torch.cuda.device_count()
        ), f"Device_id: {device_id} invalid!"
        assert grad_scalar_update_factor > 0, "grad scalar should greater than zero"
        assert (
            self.model is not None
        ), "Model not found, please init context with pytorch model"

        torch.cuda.set_device(device_id)

        world_size = env.get_world_size()
        assert world_size != -1, "WORLD_SIZE not set"
        rank_id = env.get_rank()
        assert rank_id != -1, "RANK not set"

        self.world_size = world_size
        self.rank_id = rank_id

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
                device_id,
                master_addr=master_addr,
                optimizer=dense_optimizer,
            )
            self.model = model
            _logger.info("Distributed training context init done.")
        else:
            _logger.info("SingleMachine training context init done.")

        self.dense_optimizer = dense_optimizer
        self.sparse_optimizer = sparse_optimizer

        self.wait_servers_ready()

        if lookup_emb_directly:
            init_rpc_client_num = self._init_middlewrae_rpc_client()
            _logger.info(f"Successfully init {init_rpc_client_num} rpc client")

        self.device_id = device_id

        self.update_times = 0
        self.grad_scalar_update_factor = grad_scalar_update_factor
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.backward_workers_size = backward_workers_size

        from persia.prelude import PyBackward

        self.grad_queue = Queue(grad_update_buffer_size)
        self.backward_engine = PyBackward(backward_buffer_size)

    def _enter(self):
        super()._enter()

        self.sparse_optimizer.apply()
        self.backward_engine.launch(self.device_id, self.backward_workers_size)

    def _exit(self):
        super()._exit()

        self.backward_engine.shutdown()

    @retry(wait_fixed=2000)
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

    @retry(wait_fixed=2000)
    def _init_middlewrae_rpc_client(self) -> int:
        middleware_addr_list = self.common_context.get_middleware_addr_list()
        assert len(middleware_addr_list) > 0, "Not available middleware."
        for middleware_addr in middleware_addr_list:
            self.common_context.init_rpc_client_with_addr(middleware_addr)
        return len(middleware_addr_list)

    @retry(wait_fixed=2000)
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

        loss = self.grad_scaler.scale(loss)
        scale = self.grad_scaler.get_scale()

        loss.backward()

        finite = self._on_backward(scale, embedding_gradient_check_frequency)

        self.grad_scaler.step(self.dense_optimizer)

        if finite:
            self.grad_scaler.update()
        else:
            self.grad_scaler.update(scale / self.grad_scalar_update_factor)

        self.dense_optimizer.zero_grad()

        return loss

    def _on_backward(self, loss_scale: float, embedding_gradient_check_frequency: int):
        """Update the embeddings' gradients

        Arguments:
            loss_scale (float): The loss that scaled by GradScalar.
            embedding_gradient_check_frequency (int): The frequency to check gradient finite or not for current embedding.
        """
        if self.grad_queue.full():
            self.grad_queue.get()

        finite = True
        if self.update_times % embedding_gradient_check_frequency == 0:
            finite = _check_finite(
                [emb[-1].grad for emb in self.current_batch.emb_tensors]
            )
            self.update_times += 1

        grad_slots, empty_grads = [], []
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

        torch.cuda.synchronize()
        self.backward_engine.update_sparse_gradient_batched(gradient_batch)
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
        middleware_addrs: List[str],
        *args,
        **kwargs,
    ):
        """
        Arguments:
            middleware_addrs (List[str]): middleware address(ip:port) list.
        """
        super(InferCtx, self).__init__(PreprocessMode.INFERENCE, *args, **kwargs)

        for addr in middleware_addrs:
            self.common_context.init_rpc_client_with_addr(addr)
