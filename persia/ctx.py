from enum import Enum
from queue import Queue
from typing import List, Tuple, Optional

import torch

from persia.logger import get_default_logger
from persia.sparse.optim import Optimizer
from persia.env import get_world_size, get_rank
from persia.backend import init_backend
from persia.prelude import PyPersiaReplicaInfo
from persia.error import PersiaRuntimeException
from persia.data import StreamingDataset

_logger = get_default_logger()


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

    Arguments:  # TODO: move this to __init__ doc string
        ctx_status: represent current context status # TODO: the preprocess mode to use
        block_when_exit: whether block the process when exit the context  # TODO: why?
        catch_exception: catch the exception or not when occur the exception # TODO: why?
    """

    def __init__(
        self,
        ctx_status: CtxStatus,
        block_when_exit: bool = True,
        catch_exception: bool = False,
    ):
        self.status = ctx_status
        self.block_when_exit = block_when_exit
        self.catch_exception = catch_exception

        self.current_batch = None

    @property
    def is_training(self) -> bool:
        """Whether current context is training status or not"""  # TODO:
        return self.status == CtxStatus.TRAIN

    @property
    def is_eval(self) -> bool:
        """Whether current context is eval status or not"""  # TODO:
        return self.status == CtxStatus.EVAL

    @property
    def is_inference(self) -> bool:
        """Whether current context is inference status or not"""  # TODO:
        return self.status == CtxStatus.INFERENCE

    def train(self):
        """set current context status to train"""  # TODO:
        self.status = CtxStatus.TRAIN

    def eval(self):
        """set current context status to eval"""  # TODO:
        self.status = CtxStatus.EVAL

    def inference(self):
        """set current context status to inference"""  # TODO:
        self.status = CtxStatus.INFERENCE

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, trace):
        if exc_type:
            import traceback

            _logger.error("\n" + traceback.format_exc())

        if self.block_when_exit:
            from persia.utils import block

            block()
        return PersiaRuntimeException(value)

    def prepare_features(
        self, batch: "PythonTrainBatch"
    ) -> Tuple[Tuple[torch.Tensor, List[torch.Tensor]], Optional[torch.Tensor]]:
        """Preprocess the PythonTrainBatch to PyTorch tensor.

        Arguments: batch (PythonTrainBatch): Training data provided by PersiaML
            upstream including dense, target, sparse data and meta info.

        Returns:
            # TODO: fill this

        """
        import persia_torch_ext as pte  # pytype: disable=import-error

        if self.is_inference:
            batch.target_tensor = None
        else:
            batch.target = batch.consume_all_targets()
            assert len(batch.target) == 1
            batch.target = batch.target[0]

            batch.target_tensor = pte.ptr_to_tensor_f32(
                batch.target.data_ptr(), batch.target.shape(), False
            )

        is_training = self.is_training  # cache property

        batch.dense = batch.consume_all_dense_features()
        batch.dense = batch.dense[0]
        batch.dense_tensor = pte.ptr_to_tensor_f32(
            batch.dense.data_ptr(), batch.dense.shape(), False
        )

        batch.emb = batch.consume_all_sparse_features()
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

        return (batch.dense_tensor, batch.forward_tensors), batch.target_tensor


class TrainCtx(BaseCtx):
    r"""Training context that provides full hybrid training support, including embedding
    optimizer registration, embedding lookup and update, and checkpointing.

    Example:
        # TODO: fill me

    # TODO: move arguments to __init__ doc string
    Arguments:
        mixed_precision (bool, optional): whether to enable mixed precision training
        emb_initialization (Tuple[float, float], optional): embedding uniform initialization arguments # TODO: need explanation
        admit_probability: The probability (0<=, <=1) of admitting a new embedding.
        sparse_optimizer (persia.sparse.optim.Optimizer, optional): Optimizer for the embeddings.
        weight_bound (float, optional): Restrict each element value of an embedding in [-weight_bound, weight_bound].
        ctx_status (CtxStatus, optional): represent current context status # TODO:
        device_id (int, optional): The CUDA device to use for this process.
        enable_backward (bool, optional): whether to enable embedding gradients update # TODO: training mode should always update embeddings?
        backend_worker_size (int, optional): PersiaML background thread pool size.
        backward_buffer_size (int, optional): Max number of not updated gradients queued.
        recv_buffer_size (int, optional): buffer size that recv data from data compose
        grad_queue_slot_num (int, optional): the slot size of queue that cache the torch gradient tensor to ensure
            the garbage collector collect the device memory after update_sparse_gradient_batched finished # TODO: need user understandable explanation and name
        num_backward_workers (int, optional): Number of workers sending embedding gradients in parallel.
        embedding_checkpoint(str, optional): pretrained embedding directory, load checkpoint in this dir when enter TrainCtx. # TODO: rename to checkpoint_dir, and save both dense and embedding to the same location

    """

    def __init__(
        self,
        mixed_precision: bool = True,
        emb_initialization: Tuple[float, float] = (-0.01, 0.01),
        admit_probability: float = 1.0,
        sparse_optimizer: Optional[Optimizer] = None,
        weight_bound: float = 10,
        ctx_status: CtxStatus = CtxStatus.TRAIN,
        device_id: int = 0,
        enable_backward: bool = True,
        backend_worker_size: int = 20,
        backward_buffer_size: int = 10,
        recv_buffer_size: int = 50,
        grad_queue_slot_num: int = 60,
        num_backward_workers: int = 8,
        embedding_checkpoint: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super(TrainCtx, self).__init__(ctx_status, *args, **kwargs)

        assert not self.is_training or (
            self.is_training and sparse_optimizer is not None
        ), "sparse_optimizer should not be none when is_training set to true"
        assert (
            0 <= device_id < torch.cuda.device_count()
        ), f"device_id: {device_id} invalid!"

        torch.cuda.set_device(device_id)

        self.device_id = device_id
        self.mixed_precision = mixed_precision
        self.grad_scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        self.batch_idx = 0
        self.sparse_optimizer = sparse_optimizer
        self.admit_probability = admit_probability
        self.weight_bound = weight_bound
        self.emb_initialization = emb_initialization

        world_size, rank_id = get_world_size(), get_rank()
        self.replica_info = PyPersiaReplicaInfo(world_size, rank_id)

        self.num_backward_workers = num_backward_workers

        # TODO: PyPersiaBatchFlowNatsStubResponder should decouple with data channel
        self.streaming_dataset = StreamingDataset(
            recv_buffer_size, self.replica_info
        )

        _logger.info(f"world size: {world_size} rank_id: {rank_id} init backend...")
        self.backend = init_backend(backend_worker_size, self.replica_info)
        self.enable_backward = enable_backward

        self.embedding_checkpoint = embedding_checkpoint
        self.pretrained_loaded = False

        if self.is_training or self.enable_backward:
            from persia.prelude import PyBackward

            # dynamic import the PyForward due to conditional compilation
            self.grad_queue = Queue(grad_queue_slot_num)
            self.backward_engine = PyBackward(backward_buffer_size)

    def backward(self, loss: torch.Tensor) -> torch.Tensor:

        if self.mixed_precision:
            loss = self.scaler.scale(loss).
            scale = self.scaler.get_scale()
        else:
            scale = 1

        loss.backward()
        finite = self.on_after_backward(scale)

        if self.mixed_precision:
            if finite:
                scaler.update()
            else:
                scaler.update(scale / 4.0)

        return loss

    def get_streaming_dataset(
        self,
    ) -> StreamingDataset:
        """Get streaming dataset"""  # TODO: need user understandable explanation and name or hide this if this is not expected to be used by user
        return self.streaming_dataset

    def __enter__(self):
        self.backend.set_configuration(
            self.emb_initialization[0],
            self.emb_initialization[1],
            self.admit_probability,
            self.weight_bound > 0,
            self.weight_bound,
        )
        self.sparse_optimizer.apply()

        if not self.pretrained_loaded and self.embedding_checkpoint is not None:
            self.load_embedding(self.embedding_checkpoint)
            self.pretrained_loaded = True

        if self.is_training:
            self.backward_engine.launch(self.device_id, self.num_backward_workers)

        return self

    def on_after_backward(
        self, loss_scale: float, emb_grad_check_interval: int = 20
    ):
        """Sparse embedding gradient update step that process the raw embedding and summation embedding
        gradient from raw format to standard format # TODO: cannot understand

        Arguments:
            loss_scale (float): half training loss scale to scale the gradient # TODO: this can be done without user explicitly pass in?
            batch_idx (int): index of batch data to decide the GradScalar update # TODO: cannot understand
            emb_grad_check_interval (int, optional): Gradient check interval to control the GradScalar update frequency # TODO: cannot understand
        """
        if self.grad_queue.full():
            self.grad_queue.get()

        finite = True
        if self.batch_idx % emb_grad_check_interval == 0:
            self.batch_idx += 1
            finite = _check_finite(
                [emb[-1].grad for emb in self.current_batch.emb_tensors]
            )

        grad_slot = []
        empty_grad = []
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
                empty_grad.append(emb_name)
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
                    grad_slot.append(grad)
                    gradient_batch.add_gradient(
                        emb_name,
                        grad.data_ptr(),
                        grad.shape,
                        is_f16_gradient,
                        loss_scale,
                    )

        torch.cuda.synchronize()
        self.backward_engine.update_sparse_gradient_batched(gradient_batch)
        self.grad_queue.put(grad_slot)

        if self.is_training and len(empty_grad) > 0:
            _logger.warning(
                f"current batch grad empty num: {len(empty_grad)}, {empty_grad}"
            )
        return finite

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
