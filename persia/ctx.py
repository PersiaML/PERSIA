import os

from queue import Queue
from typing import List, Tuple

import torch
from torch.utils.data import IterableDataset

from persia.logger import get_default_logger
from persia.sparse.optim import Optimizer
from persia.backend import init_backend
from persia.prelude import (
    PyPersiaReplicaInfo,
    PyPersiaBatchFlowNatsStubResponder,
    PyPersiaRpcClient,
)
from persia.error import PersiaRuntimeException
from persia.data import InfiniteIterator
from persia.service import get_middleware_services

grad_queue_slot_num = os.environ.get("GRAD_SLOT", 60)
grad_queue = Queue(grad_queue_slot_num)

logger = get_default_logger()


def _check_finite(grads: List[torch.Tensor]):
    """check all gradient tensor is finite or not"""
    return all([torch.isfinite(t).all() if t is not None else True for t in grads])


class BaseCtx:
    r"""provide feature to inject to current training step
        - half training
        - dataloder device memory shared
        - feature space fusion
    provide handy debug ctx mode debug mode for user

    Arguments:
        is_training (bool): current context is in training or not
        block_when_exit (bool): whether block the process when exit the contxt
        catch_exception (bool): catch the exception or not when occur the exception
    """

    def __init__(
        self,
        is_training: bool = False,
        block_when_exit: bool = True,
        catch_exception: bool = False,
    ):
        self.is_training = is_training
        self.block_when_exit = block_when_exit
        self.catch_exception = catch_exception

    def train(self):
        """set current context is_training to true"""
        self.is_training = True

    def eval(self):
        """set current context is_training to false"""
        self.is_training = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, trace):
        if exc_type:
            import traceback

            logger.error("\n" + traceback.format_exc())

        if self.block_when_exit:
            from persia.utils import block

            block()
        return PersiaRuntimeException(value)

    def prepare_features(self, batch, is_training=True):
        """preprocess the PythonTrainBatch
        - convert the dense, target tensor to torch float tensors
        - convert the summation embedding to float16 tensors
        - extend the raw embedding from 2D data tensor and 2D index tensor to 3D fixed size tensor, provide the corresponding
            mask to distinct the fixed position

        Arguments:
            batch (PythonTrainBatch): persia training batch data include dense, target, sparse data and meta info
            is_training (bool): whether in training scence
        """
        import persia_torch_ext as pte  # pytype: disable=import-error

        if is_training:
            batch.target = batch.consume_all_targets()
            assert len(batch.target) == 1
            batch.target = batch.target[0]

            batch.target_tensor = pte.ptr_to_tensor_f32(
                batch.target.data_ptr(), batch.target.shape(), False
            )
        else:
            batch.target_tensor = None

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
        return batch


class LocalCtx(BaseCtx):
    def __init__(self):
        ...


class TrainCtx(BaseCtx):
    r"""Training context that provide full feature of sparse training, include half training, optimzier
    register, forward, backward process

    Arguments:
        grad_scaler (torch.cuda.amp.GradScaler): scale the loss from float32 to half to support half training
        emb_initialization (Tuple[float, float]): embedding uniform initialization arguments
        admit_probability (float): embedding gradient update admit probability, range in [0, 1].
        sparse_optimizer (persias.sparse.optim.Optimizer): sparse optimizer to make embedding update available
        weight_bound (float): embedding value bound, normal will update the embedding locate in [-weight_bound, weight_bound]
        is_training (bool): current context is in training or not
        device_id (int): current cuda device id
        enable_backward (bool): enable embeddign gradients update
        backend_worker_size (int): rpc client thread pool size
        forward_buffer_size (int): forward engine buffer size
        nats_recv_buffer_size (int): nats recv buffer size, a buffer before rectifier
        backward_buffer_size (int): backward update buffer size
        rank_id (int): rank id of this process.
        world_size (int): world size of this cluster.
        num_forward_workers (int): worker num of sending forward request to servers.
        num_backward_workers (int): worker num of sending backward request to servers.
        embedding_checkpoint(str): initial embedding dir, load checkpoint in this dir when enter TrainCtx.
    """

    def __init__(
        self,
        grad_scaler: torch.cuda.amp.GradScaler = None,
        emb_initialization: Tuple[float, float] = (-0.01, 0.01),
        admit_probability: float = 1.0,
        sparse_optimizer: Optimizer = None,
        weight_bound: float = 10,
        is_training: bool = True,
        device_id: int = 0,
        enable_backward: bool = True,
        backend_worker_size: int = 20,
        forward_buffer_size: int = 10,
        nats_recv_buffer_size: int = 50,
        backward_buffer_size: int = 10,
        rank_id: int = 0,
        world_size: int = 1,
        num_forward_workers: int = 8,
        num_backward_workers: int = 8,
        embedding_checkpoint: str = None,
        *args,
        **kwargs,
    ):
        super(TrainCtx, self).__init__(is_training, *args, **kwargs)
        assert not is_training or (
            is_training and sparse_optimizer is not None
        ), "sparse_optimizer should not be none when is_training set to true"
        assert (
            0 <= device_id < torch.cuda.device_count()
        ), f"device_id: {device_id} invalid!"

        torch.cuda.set_device(device_id)

        self.device_id = device_id
        self.grad_scaler = grad_scaler
        self.is_training = is_training

        self.sparse_optimizer = sparse_optimizer
        self.admit_probability = admit_probability
        self.weight_bound = weight_bound
        self.emb_initialization = emb_initialization
        self.replica_info = PyPersiaReplicaInfo(world_size, rank_id)

        self.num_forward_workers = num_forward_workers
        self.num_backward_workers = num_backward_workers

        self.embedding_checkpoint = embedding_checkpoint
        self.pretrained_loaded = False

        # dynamic import the PyForward and PyBackward due to conditional compilation
        from persia.prelude import PyForward, PyBackward

        self.forward_engine = PyForward(
            forward_buffer_size,
            nats_recv_buffer_size,
            self.is_training,
            self.replica_info,
        )

        self._responder = PyPersiaBatchFlowNatsStubResponder(
            self.replica_info, self.forward_engine.get_input_channel()
        )

        self.backend = init_backend(backend_worker_size, self.replica_info)
        self.enable_backward = enable_backward

        if self.is_training or self.enable_backward:
            # create the backward pipeline
            self.backward_engine = PyBackward(backward_buffer_size)

        self.current_batch = None

    def data_loader(
        self,
        disorder_tolerance: float = 1.0,
        timeout: int = 1000 * 60 * 10,
    ) -> IterableDataset:
        """dataloader for fetch training data or inference data

        Arguments:
            disorder_tolerance (f32, 0.0 to 1.0): degree of orderly of dataflow, bigger means data comes more disorderly。
            timeout (int): timeout for data fetch
        """
        return InfiniteIterator(
            self.forward_engine, disorder_tolerance, timeout, self.num_forward_workers
        )

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

    def prepare_features(self, batch, is_training=True):
        """preprocess the PythonTrainBatch
        - convert the dense, target tensor to torch float tensors
        - convert the summation embedding to float16 tensors
        - extend the raw embedding from 2D data tensor and 2D index tensor to 3D fixed size tensor, provide the corresponding
            mask to distinct the fixed position

        Arguments:
            batch (PythonTrainBatch): persia training batch data include dense, target, sparse data and meta info
            is_training (bool): whether in training scence
        """
        batch = super().prepare_features(batch, is_training)
        self.current_batch = batch
        return (batch.dense_tensor, batch.forward_tensors), batch.target_tensor

    def on_after_backward(
        self, loss_scale: float, batch_idx: int, emb_grad_check_interval: int = 20
    ):
        """Sparse embedding gradient update step that process the raw embedding and summation embedding
        gradient from raw format to standar format

        Arguments:
            loss_scale (float): half training loss scale to scale the gradient
            batch_idx (int): index of batch data to decide the GradScalar update
            emb_grad_check_interval (int): check interval to controll the GradScalar update frequnency
        """
        if grad_queue.full():
            grad_queue.get()

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
        grad_queue.put(grad_slot)
        if self.is_training and len(empty_grad) > 0:
            logger.warning(
                f"current batch grad empty num: {len(empty_grad)}, {empty_grad}"
            )
        return finite

    def dump_embedding(self, dst_dir: str, blocking: bool = False):
        self.backend.dump_embedding(dst_dir, blocking)

    def load_embedding(self, dst_dir: str, blocking: bool = True):
        self.backend.load_embedding(dst_dir, blocking)

    def wait_for_dump_embedding(self):
        self.backend.wait_for_dump_embedding()

    def wait_for_load_embedding(self):
        self.backend.wait_for_load_embedding()


class InferCtx(BaseCtx):
    r"""Inference context that provide full feature of embedding lookup

    Arguments:
        backend_worker_size (int): rpc client thread pool size
        middleware_addr([str]): addr of middlewares

    """

    def __init__(
        self,
        backend_worker_size: int = 20,
        middleware_addrs: str = None,
        *args,
        **kwargs,
    ):
        super(InferCtx, self).__init__(False, *args, **kwargs)
        self.rpc_client = PyPersiaRpcClient(backend_worker_size)
        if middleware_addrs is None:
            middleware_addrs = get_middleware_services()
        for addr in middleware_addrs:
            self.rpc_client.add_rpc_client(addr)

    def prepare_features(self, batch):
        """prepare data for inference

        Arguments:
            batch (PythonTrainBatch): persia training batch data include dense, target, sparse data and meta info
        """
        batch = super().prepare_features(batch, False)
        return (batch.dense_tensor, batch.forward_tensors)
