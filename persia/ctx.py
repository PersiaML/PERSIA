import os

from enum import Enum
from queue import Queue
from typing import List, Tuple, Optional, NewType

import torch

import persia.env as env
from persia.logger import get_default_logger
from persia.sparse.optim import Optimizer
from persia.backend import init_backend
from persia.prelude import PyPersiaReplicaInfo
from persia.error import PersiaRuntimeException

_CURRENT_CXT = None

_logger = get_default_logger()

PythonTrainBatch = NewType("PythonTrainBatch", object)


def _check_finite(tensors: List[torch.Tensor]) -> bool:
    """Check all tensors in the input list contain only finite elements.

    Arguments:
        tensors: tensor list that need to check finite or not
    """
    return all([torch.isfinite(t).all() if t is not None else True for t in tensors])


class PreprocessMode(Enum):
    r"""Enum struct to identify which status current context is.
    Context will preprocess the batch input data according to different status.
    # TODO: need more explanation about each status
    """
    TRAIN = 1
    EVAL = 2
    INFERENCE = 3


class BaseCtx:
    r"""Base context to provide fundamental function # TODO: too vague, need more explanation about what this does and where to use

    Examples:
        with BaseCtx() as ctx:
            ...
    """

    def __init__(
        self,
        backend_worker_size: int = 10,
        block_when_exit: bool = True,
        catch_exception: bool = False,
    ):
        """
        Arguments:
            backend_worker_size (int): Rpc ThreadPool worker size
            block_when_exit (bool): whether block the process when exit the context  # TODO: why?
            catch_exception (bool): catch the exception or not when occur the exception # TODO: why?

        """
        self.block_when_exit = block_when_exit
        self.catch_exception = catch_exception

        world_size = env.get_world_size()

        if world_size == -1:
            replica_size = env.get_replica_size()
            replica_index = env.get_replica_index()
            self.replica_info = PyPersiaReplicaInfo(replica_size, replica_index)
            _logger.info(
                f"init datacompose backend replica_size: {replica_size} replica_index: {replica_index}"
            )
        else:
            rank_id = env.get_rank()
            self.replica_info = PyPersiaReplicaInfo(world_size, rank_id)
            _logger.info(
                f"init trainer backend world size: {world_size} rank_id: {rank_id}"
            )

        self.backend = init_backend(backend_worker_size, self.replica_info)

    def _enter(self):
        """Hook when enter the context"""
        ...

    def _exit(self):
        """Hook when exit the context"""
        ...

    def __enter__(self):
        self._enter()

        global _CURRENT_CXT
        _CURRENT_CXT = self

        return self

    def __exit__(self, exc_type, value, trace):
        self._exit()

        global _CURRENT_CXT
        _CURRENT_CXT = None

        if exc_type:
            import traceback

            _logger.error("\n" + traceback.format_exc())

        if self.block_when_exit:
            from persia.utils import block

            block()
        return PersiaRuntimeException(value)


class EmbeddingCtx(BaseCtx):
    r"""This class provide the embedding relative function compare to BaseCtx.It can provide the basic offline test or inference
    according to different preprocess_mode.The simple way to use this context is call ``persia.ctx.eval_ctx()`` or ``persia.ctx.inference_ctx``
    to get the context

    Examples:
        ...

    """

    def __init__(
        self,
        preprocess_mode: PreprocessMode,
        emb_initialization: Tuple[float, float] = (-0.01, 0.01),
        admit_probability: float = 1.0,
        weight_bound: float = 10,
        embedding_checkpoint: Optional[str] = None,
        *args,
        **kwargs,
    ):

        """
        Arguments:
            preprocess_mode (PreprocessMode): Preprcess mode of current ctx
            emb_initialization (Tuple[float, float], optional): embedding uniform initialization arguments # TODO: need explanation
            admit_probability (float, optional): The probability (0<=, <=1) of admitting a new embedding.
            weight_bound (float, optional): Restrict each element value of an embedding in [-weight_bound, weight_bound].
            backend_worker_size (int, optional): PersiaML background thread pool size.
            recv_buffer_size (int, optional): buffer size that recv data from data compose
            grad_queue_slot_num (int, optional): the slot size of queue that cache the torch gradient tensor to ensure
                the garbage collector collect the device memory after update_sparse_gradient_batched finished # TODO: need user understandable explanation and name
            embedding_checkpoint(str, optional): pretrained embedding directory, load checkpoint in this dir when enter TrainCtx. # TODO: rename to checkpoint_dir, and save both dense and embedding to the same location
        """
        super(EmbeddingCtx, self).__init__(*args, **kwargs)
        self.preprocess_mode = preprocess_mode
        self.emb_initialization = emb_initialization
        self.admit_probability = admit_probability
        self.weight_bound = weight_bound
        self.embedding_checkpoint = embedding_checkpoint

        self.current_batch = None
        self.pretrained_loaded = False

    def _enter(self):
        self.backend.set_configuration(
            self.emb_initialization[0],
            self.emb_initialization[1],
            self.admit_probability,
            self.weight_bound > 0,
            self.weight_bound,
        )

        if not self.pretrained_loaded and self.embedding_checkpoint is not None:
            self.load_embedding(self.embedding_checkpoint)
            self.pretrained_loaded = True

    def prepare_features(
        self, batch: PythonTrainBatch
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
        """Preprocess the PythonTrainBatch to PyTorch tensor.

        Arguments:
            batch (PythonTrainBatch): Training data provided by PersiaML upstream including
                dense, target, sparse data and meta info.

        Returns:
            return the Tuple,
            # TODO: fill this
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

    def dump_cpk(
        self,
        dst_dir: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        default_filename: str = "dense.pt",
        blocking: bool = True,
    ):
        """Dump the dense and sparse ckp to destination directory.

        Arguments:
            dst_dir (str): Destination directory.
            model (torch.nn.Module, optional): Pytorch model instance.
            optimizer (torch.optim.Optimizer, optional): Pytorch optimizer instance.
            default_filename (str, optional): Dense checkpoint filename.
            blocking (bool, optional): Dump embedding checkpoint in blocking mode or not.
        """
        os.makedirs(dst_dir, exist_ok=True)

        cpk_dict = {}
        if model:
            cpk_dict["model"] = model.state_dict()

        if optimizer:
            cpk_dict["opt"] = optimizer.state_dict()

        if len(cpk_dict.keys()) > 0:
            dense_model_filepath = os.path.join(dst_dir, default_filename)
            torch.save(cpk_dict, dense_model_filepath)

        self.dump_embedding(dst_dir, blocking=blocking)

    def load_cpk(
        self,
        src_dir: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None,
        default_filename: str = "dense.pt",
        blocking: bool = True,
    ):
        """Load the dense and sparse ckp from source directory.

        Arguments:
            src_dir (str): Source directory.
            model (torch.nn.Module, optional): Pytorch model instance.
            optimizer (torch.optim.Optimizer, optional): Pytorch optimizer instance.
            map_location (str, optional): Load the dense checkpoint to specific device.
            default_filename (str, optional): Dense checkpoint filename.
            blocking (bool, optional): Dump embedding checkpoint in blocking mode or not.
        """
        if not os.path.exists(src_dir):
            _logger.warn(f"src_dir: {src_dir} not exists")
            return

        dense_model_filepath = os.path.join(src_dir, default_filename)
        if os.path.exists(dense_model_filepath):
            cpk_dict = torch.load(dense_model_filepath, map_location)

            if model:
                model.load_state_dict(cpk_dict["model"], strict=True)

            if optimizer:
                optimizer.load_state_dict(cpk_dict["opt"])

        self.load_embedding(src_dir, blocking=blocking)

    def dump_embedding(self, dst_dir: str, blocking: bool = False):
        """Dump embeddings to ``dst_dir``. Use ``TrainCtx.wait_for_dump_embedding`` to wait until finished
        if ``blocking=False``.

        Arguments:
            dst_dir (str): Destination directory.
            blocking (bool, optional): Dump embedding in blocking mode or not.
        """
        self.backend.dump_embedding(dst_dir, blocking)

    def load_embedding(self, src_dir: str, blocking: bool = True):
        """Load embeddings from ``src_dir``. Use ``TrainCtx.wait_for_load_embedding`` to wait until finished
        if ``blocking=False``.

        Arguments:
            src_dir (str): Directory to load embeddings.
            blocking (bool, optional): Dump embedding in blocking mode or not.
        """
        self.backend.load_embedding(src_dir, blocking)

    def wait_for_dump_embedding(self):
        """Wait for the embedding dump process."""
        self.backend.wait_for_dump_embedding()

    def wait_for_load_embedding(self):
        """Wait for the embedding load process."""
        self.backend.wait_for_load_embedding()


class TrainCtx(EmbeddingCtx):
    r"""Training context that provides full hybrid training support, including embedding
    optimizer registration, embedding lookup and update, and checkpointing.

    Example:
        # TODO: fill me

    """

    def __init__(
        self,
        sparse_optimizer: Optimizer,
        dense_optimizer: torch.optim.Optimizer,
        device_id: int = 0,
        mixed_precision: bool = False,
        grad_scalar_update_factor: float = 4,
        backward_buffer_size: int = 10,
        num_backward_workers: int = 8,
        grad_queue_slot_num: int = 60,
        *args,
        **kwargs,
    ):
        """
        Arguments:
            sparse_optimizer (persia.sparse.optim.Optimizer): Optimizer for the embeddings.
            dense_optimizer (torch.optim.Optimizer): Optimizer for dense parameters
            device_id (int, optional): The CUDA device to use for this process.
            mixed_precision (bool, optional): whether to enable mixed precision training
            grad_scalar_update_factor (float, optional): Update factor of Gradscalar to ensure loss scale finitely if set ``mixed_precision=True``
            backward_buffer_size (int, optional): Max number of not updated gradients queued.
            num_backward_workers (int, optional): Number of workers sending embedding gradients in parallel.
        """
        super(TrainCtx, self).__init__(PreprocessMode.TRAIN, *args, **kwargs)

        assert (
            sparse_optimizer is not None
        ), "Sparse_optimizer should not be none in train context"
        assert (
            0 <= device_id < torch.cuda.device_count()
        ), f"device_id: {device_id} invalid!"

        torch.cuda.set_device(device_id)

        self.device_id = device_id

        self.batch_idx = 0
        self.mixed_precision = mixed_precision
        self.grad_scalar_update_factor = grad_scalar_update_factor
        self.grad_scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        self.sparse_optimizer = sparse_optimizer
        self.dense_optimizer = dense_optimizer
        self.num_backward_workers = num_backward_workers

        from persia.prelude import PyBackward

        # dynamic import the PyForward due to conditional compilation
        self.grad_queue = Queue(grad_queue_slot_num)
        self.backward_engine = PyBackward(backward_buffer_size)

    def _enter(self):
        super()._enter()

        self.sparse_optimizer.apply()
        self.backward_engine.launch(self.device_id, self.num_backward_workers)

    def backward(
        self, loss: torch.Tensor, emb_grad_check_interval: int = 20
    ) -> torch.Tensor:
        """Update the dense and sparse states by current loss

        Arguments:
            loss (torch.Tensor): Loss of current batch
            emb_grad_check_interval (int, optional): Gradient check interval to control the GradScalar update frequency # TODO: cannot understand
        """
        assert (
            self.dense_optimizer is not None
        ), "dense_optimizer not exists in TrainCtx"

        if self.mixed_precision:
            loss = self.grad_scaler.scale(loss)
            scale = self.grad_scaler.get_scale()
        else:
            scale = 1

        loss.backward()

        finite = self._on_after_backward(scale)

        if self.mixed_precision:
            if finite:
                self.grad_scaler.update()
            else:
                self.grad_scaler.update(scale / self.grad_scalar_update_factor)

            self.grad_scaler.step(self.dense_optimizer)

        self.dense_optimizer.zero_grad()

        return loss

    def _on_after_backward(self, loss_scale: float, emb_grad_check_interval: int = 20):
        """Update the embeddings gradients

        Arguments:
            emb_grad_check_interval (int): Interval to check the embedding gradient finite or not.
        """
        if self.grad_queue.full():
            self.grad_queue.get()

        finite = True
        if self.batch_idx % emb_grad_check_interval == 0:
            self.batch_idx += 1
            finite = _check_finite(
                [emb[-1].grad for emb in self.current_batch.emb_tensors]
            )

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


def cnt_ctx() -> Optional[BaseCtx]:
    """Get the BaseCtx recently entered"""
    return _CURRENT_CXT


def base_ctx(*args, **kwargs) -> BaseCtx:
    return BaseCtx(*args, **kwargs)


def eval_ctx(*args, **kwargs) -> EmbeddingCtx:
    return EmbeddingCtx(PreprocessMode.EVAL, *args, **kwargs)


def inference_ctx(*args, **kwargs) -> EmbeddingCtx:
    return EmbeddingCtx(PreprocessMode.INFERENCE, *args, **kwargs)
