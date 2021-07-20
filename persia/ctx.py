from enum import Enum
from queue import Queue
from typing import List, Tuple, Optional

import torch

from persia.logger import get_default_logger
from persia.sparse.optim import Optimizer
from persia.backend import init_backend
from persia.prelude import PyPersiaReplicaInfo
from persia.error import PersiaRuntimeException
from persia.data import NatsStreamingChannel

_logger = get_default_logger()


def _check_finite(grads: List[torch.Tensor]):
    """check all gradient tensor is finite or not

    Arguments:
        grads (List[torch.Tensor]): grad tensor list that need to check finite or not
    """
    return all([torch.isfinite(t).all() if t is not None else True for t in grads])


class CtxStatus(Enum):
    r"""Enum struct to identify which status current context is.Context will preprocess the
    batch input data according to different status.
    """
    TRAIN = 1
    EVAL = 2
    INFERENCE = 3


class BaseCtx:
    r"""Base context to provide fundamental function

    Arguments:
        ctx_status (CtxStatus): represent current context status
        block_when_exit (bool, optional): whether block the process when exit the contxt
        catch_exception (bool, optional): catch the exception or not when occur the exception
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
        """Whether current context is training status or not"""
        return self.status == CtxStatus.TRAIN

    @property
    def is_eval(self) -> bool:
        """Whether current context is eval status or not"""
        return self.status == CtxStatus.EVAL

    @property
    def is_inference(self) -> bool:
        """Whether current context is inference status or not"""
        return self.status == CtxStatus.INFERENCE

    def train(self):
        """set current context status to train"""
        self.status = CtxStatus.TRAIN

    def eval(self):
        """set current context status to eval"""
        self.status = CtxStatus.EVAL

    def inference(self):
        """set current context status to inference"""
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

    def prepare_features(self, batch):
        """preprocess the PythonTrainBatch to pytorch tensor. Multiple dense, sparse and target tensors
        will converted to tensor from rust cuda pointer.

        Arguments:
            batch (PythonTrainBatch): persia training batch data include dense, target, sparse data and meta info
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
        # assert len(batch.dense) == 1
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
    r"""Training context that provide full feature of sparse training, include optimzier
    register, embedding forward, embedding backward process, ckp load and ckp dump.

    Arguments:
        grad_scaler (torch.cuda.amp.GradScaler, optional): scale the loss from float32 to half to support half training
        emb_initialization (Tuple[float, float], optional): embedding uniform initialization arguments
        admit_probability (float): probability of embedding generation. value in [0, 1]. Generate a random
            value(range in [0, 1]) for each sparse embedding, generate the new embedding once the value is
            small than the admit_probability. Always generate the new embedding when the admit_probability set to 1
        sparse_optimizer (persias.sparse.optim.Optimizer, optional): sparse optimizer to make embedding update available
        weight_bound (float, optional): embedding value bound, normal will update the embedding locate in [-weight_bound, weight_bound]
        ctx_status (CtxStatus, optional): represent current context status
        device_id (int, optional): current cuda device id
        enable_backward (bool, optional): enable embeddign gradients update
        backend_worker_size (int, optional): rpc client thread pool size
        backward_buffer_size (int, optional): backward update buffer size
        nats_recv_buffer_size (int, optional): nats recv buffer size, a buffer before rectifier
        grad_queue_slot_num (int, optional): the slot size of queue that cache the torch gradient tensor to ensure
            the garbage collector collect the device memory after update_sparse_gradient_batched finished
        rank_id (int, optional): rank id of this process.
        world_size (int, optional): world size of this cluster.
        num_backward_workers (int, optional): worker num of sending backward request to servers.
        embedding_checkpoint(str, optional): pretrained embedding directory, load checkpoint in this dir when enter TrainCtx.
    """

    def __init__(
        self,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
        emb_initialization: Tuple[float, float] = (-0.01, 0.01),
        admit_probability: float = 1.0,
        sparse_optimizer: Optional[Optimizer] = None,
        weight_bound: float = 10,
        ctx_status: CtxStatus = CtxStatus.TRAIN,
        device_id: int = 0,
        enable_backward: bool = True,
        backend_worker_size: int = 20,
        backward_buffer_size: int = 10,
        nats_recv_buffer_size: int = 50,
        grad_queue_slot_num: int = 60,
        rank_id: int = 0,
        world_size: int = 1,
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
        self.grad_scaler = grad_scaler

        self.sparse_optimizer = sparse_optimizer
        self.admit_probability = admit_probability
        self.weight_bound = weight_bound
        self.emb_initialization = emb_initialization

        self.replica_info = PyPersiaReplicaInfo(world_size, rank_id)

        self.num_backward_workers = num_backward_workers
        # TODO: PyPersiaBatchFlowNatsStubResponder should decouple with data channel
        # data channel create by datachannel
        self.nats_dataset = NatsStreamingChannel(
            nats_recv_buffer_size, self.replica_info
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

    def get_nats_streaming_datachannel(
        self,
    ) -> NatsStreamingChannel:
        """Get nats streaming datachannel"""
        return self.nats_dataset

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
        self, loss_scale: float, batch_idx: int, emb_grad_check_interval: int = 20
    ):
        """Sparse embedding gradient update step that process the raw embedding and summation embedding
        gradient from raw format to standar format

        Arguments:
            loss_scale (float): half training loss scale to scale the gradient
            batch_idx (int): index of batch data to decide the GradScalar update
            emb_grad_check_interval (int, optional):Gradient check interval to controll the GradScalar update frequnency
        """
        if self.grad_queue.full():
            self.grad_queue.get()

        finite = True
        if batch_idx % emb_grad_check_interval == 0:
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
        """Dump the sparse embedding to destination directory.This method provide not blocking
        mode, invoke the TrainCtx.wait_for_dump_embedding once set blocking to False.

        Arguments:
            dst_dir (str): destination directory
            blocking (bool, optional): dump embedding in blocking mode or not
        """
        self.backend.dump_embedding(dst_dir, blocking)

    def load_embedding(self, src_dir: str, blocking: bool = True):
        """Load the sparse embedding from source directory.This method provide not blocking mode,
        invoke the TrainCtx.wait_for_load_embedding once set the blocking to False.

        Arguments:
            src_dir (str): destination directory
            blocking (bool, optional): dump embedding in blocking mode or not
        """
        self.backend.load_embedding(src_dir, blocking)

    def wait_for_dump_embedding(self):
        """Wait for dump the sparse embedding"""
        self.backend.wait_for_dump_embedding()

    def wait_for_load_embedding(self):
        """Wait for load the sparse embedding"""
        self.backend.wait_for_load_embedding()
