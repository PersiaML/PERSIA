use crate::backward::PythonGradientBatch;
use crate::cuda::utils::{cuda_dense_tensor_h2d, embedding2cuda_tensor};
use crate::cuda::{set_device, AsyncEmbeddingOnCuda, AsyncTensorOnCuda};
use crate::metrics::MetricsHolder;
use crate::utils::PyPersiaBatchDataReceiver;
use crate::PersiaCommonContext;

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use persia_common::{EmbeddingBatch, EmbeddingTensor, PersiaBatchData, SparseBatchRemoteReference};
use persia_embedding_config::PersiaReplicaInfo;
use persia_embedding_server::middleware_service::MiddlewareServerError;
use persia_libs::{
    flume,
    tokio::{
        self,
        sync::{OwnedSemaphorePermit, Semaphore},
        task::JoinHandle as TokioJoinHandle,
    },
    tracing,
};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass]
pub struct PythonAsyncEmbeddingOnCuda {
    inner: Option<AsyncEmbeddingOnCuda>,
}

#[pymethods]
impl PythonAsyncEmbeddingOnCuda {
    pub fn is_raw_embedding(&self) -> bool {
        match self.inner.as_ref().unwrap() {
            AsyncEmbeddingOnCuda::Raw(_) => true,
            AsyncEmbeddingOnCuda::Sum(_) => false,
        }
    }
    pub fn get_sum_embedding(&mut self) -> PythonAsyncTensorOnCuda {
        if let AsyncEmbeddingOnCuda::Sum(sum_embedding) = self.inner.take().unwrap() {
            PythonAsyncTensorOnCuda {
                inner: sum_embedding,
                done_sync: false,
            }
        } else {
            panic!("AttrError: raw embedding can not convert to sum embedding")
        }
    }

    pub fn get_raw_embedding(
        &mut self,
    ) -> (
        PythonAsyncTensorOnCuda,
        PythonAsyncTensorOnCuda,
        PythonAsyncTensorOnCuda,
        Vec<usize>,
    ) {
        if let AsyncEmbeddingOnCuda::Raw(raw_embedding) = self.inner.take().unwrap() {
            (
                PythonAsyncTensorOnCuda {
                    inner: raw_embedding.tensor,
                    done_sync: false,
                },
                PythonAsyncTensorOnCuda {
                    inner: raw_embedding.index,
                    done_sync: false,
                },
                PythonAsyncTensorOnCuda {
                    inner: raw_embedding.non_empty_index,
                    done_sync: false,
                },
                raw_embedding.samples_id_num,
            )
        } else {
            panic!("AttrError: sum embedding can not convert to raw embedding")
        }
    }
}

#[pyclass]
pub struct PythonAsyncTensorOnCuda {
    inner: AsyncTensorOnCuda,
    done_sync: bool,
}

#[pymethods]
impl PythonAsyncTensorOnCuda {
    pub fn sync_event(&mut self) {
        if !self.done_sync {
            self.inner.event.synchronize();
            self.done_sync = true;
        }
    }

    pub fn name(&self) -> &str {
        self.inner.name.as_str()
    }

    pub fn data_ptr(&mut self) -> u64 {
        self.sync_event();
        self.inner.ptr.inner as u64
    }

    pub fn shape(&self) -> [usize; 2] {
        self.inner.shape
    }

    pub fn num_bytes(&self) -> usize {
        self.inner.ptr.num_bytes
    }
}

#[pyclass(dict)]
pub struct PythonTrainBatch {
    pub inner: PersiaTrainingBatchOnGpu,
}

#[pymethods]
impl PythonTrainBatch {
    pub fn middleware_server_addr(&self) -> &str {
        self.inner.middleware_server_addr.as_str()
    }

    pub fn consume_all_dense_features(&mut self) -> Vec<PythonAsyncTensorOnCuda> {
        std::mem::replace(&mut self.inner.dense, vec![])
            .into_iter()
            .map(|x| PythonAsyncTensorOnCuda {
                inner: x,
                done_sync: false,
            })
            .collect()
    }

    pub fn consume_all_sparse_features(&mut self) -> Vec<PythonAsyncEmbeddingOnCuda> {
        std::mem::replace(&mut self.inner.embeddings, vec![])
            .into_iter()
            .map(|x| PythonAsyncEmbeddingOnCuda { inner: Some(x) })
            .collect()
    }

    pub fn consume_all_targets(&mut self) -> Vec<PythonAsyncTensorOnCuda> {
        std::mem::replace(&mut self.inner.target, vec![])
            .into_iter()
            .map(|x| PythonAsyncTensorOnCuda {
                inner: x,
                done_sync: false,
            })
            .collect()
    }

    pub fn consume_all_map_data(&mut self) {}

    pub fn consume_all_meta_data<'a>(&mut self, py: Python<'a>) -> Option<&'a PyBytes> {
        if self.inner.meta_data.is_some() {
            return Some(PyBytes::new(
                py,
                self.inner.meta_data.as_ref().unwrap().as_slice(),
            ));
        } else {
            return None;
        };
    }

    pub fn create_gradient_batch(&mut self) -> PythonGradientBatch {
        PythonGradientBatch::new(
            self.inner.forward_id,
            self.inner.middleware_server_addr.as_str(),
            self.inner.embedding_staleness_permit.take(),
        )
    }
}

#[derive(Debug)]
pub struct PersiaTrainingBatchOnGpu {
    pub dense: Vec<AsyncTensorOnCuda>,
    pub embeddings: Vec<AsyncEmbeddingOnCuda>,
    pub target: Vec<AsyncTensorOnCuda>,
    pub meta_data: Option<Vec<u8>>,
    pub middleware_server_addr: String,
    pub forward_id: u64,
    pub embedding_staleness_permit: Option<OwnedSemaphorePermit>,
}

struct PerisaDataOrderManager {
    pub world_size: usize,
    pub expect_batch_id: usize,
    pub data_buffer: BinaryHeap<PersiaBatchData>,
    pub latest_pop: Instant,
}

const DATA_BUFFER_SIZE: usize = 32;

impl PerisaDataOrderManager {
    pub fn new(world_size: usize, rank_id: usize) -> Self {
        Self {
            world_size,
            expect_batch_id: rank_id,
            data_buffer: BinaryHeap::with_capacity(DATA_BUFFER_SIZE),
            latest_pop: Instant::now(),
        }
    }

    pub fn spawn_reorder_worker(
        world_size: usize,
        rank_id: usize,
        channel_r: flume::Receiver<PersiaBatchData>,
        channel_s: flume::Sender<PersiaBatchData>,
        running: Arc<AtomicBool>,
    ) -> JoinHandle<()> {
        std::thread::spawn(move || {
            let mut order_manager = Self::new(world_size, rank_id);
            loop {
                if !running.load(Ordering::Acquire) {
                    break;
                }
                if let Ok(batch) = channel_r.recv_timeout(Duration::from_millis(10)) {
                    order_manager.data_buffer.push(batch);
                    while let Some(data) = order_manager.data_buffer.peek() {
                        let data_batch_id = data.batch_id.unwrap_or(usize::MIN);
                        if data_batch_id <= order_manager.expect_batch_id {
                            if let Err(e) = channel_s.send(order_manager.pop_from_buffer().unwrap())
                            {
                                tracing::debug!("failed to send data to reorder buffer {:?}", e);
                            }
                        } else {
                            break;
                        }
                    }
                } else if order_manager.data_buffer.len() > 0
                    && order_manager.latest_pop.elapsed().as_secs() > 1
                {
                    tracing::warn!(
                            "PerisaDataOrderManager waiting for batch exceed 1 sec, it's possible because that data input is slow.
                            Now, pulling all buffered input batches, which may causes disorderly of input batch. If you do not care
                            about the order of input batches, please set shuffle to true to get a higher training efficiency."
                        );
                    while let Some(poped) = order_manager.pop_from_buffer() {
                        if let Err(e) = channel_s.send(poped) {
                            tracing::debug!("failed to send data to reorder buffer {:?}", e);
                        }
                    }
                }
            }
        })
    }

    fn pop_from_buffer(&mut self) -> Option<PersiaBatchData> {
        if let Some(poped) = self.data_buffer.pop() {
            self.latest_pop = Instant::now();
            self.expect_batch_id = poped.batch_id.unwrap_or(self.expect_batch_id) + self.world_size;
            Some(poped)
        } else {
            None
        }
    }
}

struct Forward {
    pub input_channel: Option<flume::Receiver<PersiaBatchData>>,
    pub reorder_buffer_channel_s: Option<flume::Sender<PersiaBatchData>>,
    pub reorder_buffer_channel_r: Option<flume::Receiver<PersiaBatchData>>,
    pub forwarded_channel_s: flume::Sender<(
        PersiaBatchData,
        EmbeddingBatch,
        Option<OwnedSemaphorePermit>,
    )>,
    pub forwarded_channel_r: flume::Receiver<(
        PersiaBatchData,
        EmbeddingBatch,
        Option<OwnedSemaphorePermit>,
    )>,
    pub gpu_forwarded_channel_s: flume::Sender<PersiaTrainingBatchOnGpu>,
    pub gpu_forwarded_channel_r: flume::Receiver<PersiaTrainingBatchOnGpu>,
    pub is_training: bool,
    pub launch: bool,
    pub embedding_staleness_semaphore: Option<Arc<Semaphore>>,
    pub std_handles: Vec<std::thread::JoinHandle<()>>,
    pub tokio_handles: Vec<TokioJoinHandle<()>>,
    pub running: Arc<AtomicBool>,
}

impl Forward {
    fn new(
        forward_buffer_size: usize,
        is_training: bool,
        reproducible: bool,
        embedding_staleness: Option<usize>,
    ) -> Self {
        let (reorder_buffer_channel_s, reorder_buffer_channel_r) = if reproducible {
            let (s, r) = flume::bounded(1);
            (Some(s), Some(r))
        } else {
            (None, None)
        };
        let (forwarded_channel_s, forwarded_channel_r) = flume::bounded(forward_buffer_size);
        let (gpu_forwarded_channel_s, gpu_forwarded_channel_r) =
            flume::bounded(forward_buffer_size);

        let embedding_staleness_semaphore = match embedding_staleness {
            Some(s) => Some(Arc::new(Semaphore::new(s))),
            None => None,
        };

        Self {
            input_channel: None,
            reorder_buffer_channel_s,
            reorder_buffer_channel_r,
            forwarded_channel_r,
            forwarded_channel_s,
            gpu_forwarded_channel_r,
            gpu_forwarded_channel_s,
            is_training,
            launch: false,
            embedding_staleness_semaphore,
            std_handles: Vec::new(),
            tokio_handles: Vec::new(),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn launch(&mut self, device_id: i32, num_workers: usize) -> PyResult<()> {
        if !self.launch {
            match &self.input_channel {
                Some(_) => {
                    self.running.store(true, Ordering::Relaxed);
                    if self.reorder_buffer_channel_r.is_some() {
                        self.spawn_reorder_buffer_worker()?;
                    }
                    self.spawn_to_gpu_worker(device_id);
                    self.spawn_forward_worker(num_workers);
                    self.launch = true;
                    Ok(())
                }
                None => Err(PyRuntimeError::new_err(
                    "please set input channel before launch the forward engine",
                )),
            }
        } else {
            tracing::warn!("forward engine already launch");
            Ok(())
        }
    }

    pub fn shutdown(&mut self) -> PyResult<()> {
        tracing::info!("exiting persia forward context");
        self.running.store(false, Ordering::Relaxed);
        Ok(())
    }

    fn spawn_reorder_buffer_worker(&mut self) -> PyResult<()> {
        let replica_info = PersiaReplicaInfo::get().expect("not in persia context");
        let handler = PerisaDataOrderManager::spawn_reorder_worker(
            replica_info.replica_size,
            replica_info.replica_index,
            self.input_channel.as_ref().unwrap().clone(),
            self.reorder_buffer_channel_s.as_ref().unwrap().clone(),
            self.running.clone(),
        );
        self.std_handles.push(handler);
        Ok(())
    }

    fn spawn_to_gpu_worker(&mut self, device_id: i32) {
        let channel_r = self.forwarded_channel_r.clone();
        let channel_s = self.gpu_forwarded_channel_s.clone();

        let running = self.running.clone();
        let handler = std::thread::spawn(move || {
            set_device(device_id);
            loop {
                if !running.load(Ordering::Acquire) {
                    break;
                }
                let start_time = std::time::Instant::now();
                if let Ok((batch, embeddings, embedding_staleness_permit)) = channel_r.recv() {
                    tracing::debug!("get forwarded batch time cost {:?}", start_time.elapsed());
                    let embeddings: Vec<_> = embeddings
                        .batches
                        .into_iter()
                        .map(|feature_embedding_batch| {
                            embedding2cuda_tensor(feature_embedding_batch)
                        })
                        .collect();

                    let dense_tensors: Vec<AsyncTensorOnCuda> = batch
                        .dense_data
                        .into_iter()
                        .map(|d| cuda_dense_tensor_h2d(d).expect("cannot move dense to gpu"))
                        .collect();

                    let target_tensors: Vec<AsyncTensorOnCuda> = batch
                        .target_data
                        .into_iter()
                        .map(|t| cuda_dense_tensor_h2d(t).expect("cannot move target to gpu"))
                        .collect();

                    let (middleware_addr, forward_id) = batch.sparse_data.to_forward_id();
                    let training_batch = PersiaTrainingBatchOnGpu {
                        dense: dense_tensors,
                        embeddings,
                        target: target_tensors,
                        meta_data: batch.meta_data,
                        middleware_server_addr: middleware_addr.to_string(),
                        forward_id,
                        embedding_staleness_permit,
                    };
                    if let Err(e) = channel_s.send(training_batch) {
                        tracing::debug!("failed to send data to gpu_forwarded_channel_s {:?}", e);
                    }

                    if let Ok(m) = MetricsHolder::get() {
                        m.forward_client_to_gpu_time_cost
                            .observe(start_time.elapsed().as_secs_f64());
                    }
                }
            }
        });

        self.std_handles.push(handler);
    }

    fn spawn_forward_worker(&mut self, num_workers: usize) {
        let context = PersiaCommonContext::get();
        let _guard = context.async_runtime.enter();

        for _ in 0..num_workers {
            let channel_s = self.forwarded_channel_s.clone();
            let channel_r = if let Some(reorder_channel) = self.reorder_buffer_channel_r.as_ref() {
                reorder_channel.clone()
            } else {
                self.input_channel.as_ref().unwrap().clone()
            };

            let rpc_client = context.rpc_client.clone();
            let is_training = self.is_training;

            let running = self.running.clone();
            let embedding_staleness_semaphore = match &self.embedding_staleness_semaphore {
                Some(s) => Some(s.clone()),
                None => None,
            };

            let handle = tokio::spawn(async move {
                loop {
                    if !running.load(Ordering::Acquire) {
                        break;
                    }
                    let start_time = std::time::Instant::now();
                    if let Ok(batch) = channel_r.recv_async().await {
                        tracing::debug!(
                            "get deserialized message time cost {:?}",
                            start_time.elapsed()
                        );

                        let mut batch = batch;
                        let (embeddings_rpc_result, middleware_addr, embedding_staleness_permit) =
                            match batch.sparse_data {
                                EmbeddingTensor::SparseBatch(mut sparse_data) => {
                                    let (middleware_addr, client) =
                                        rpc_client.get_random_client_with_addr();

                                    sparse_data.requires_grad = is_training;
                                    let result = client.forward_batched_direct(&sparse_data).await;

                                    (result, middleware_addr, None)
                                }
                                EmbeddingTensor::SparseBatchRemoteReference(sparse_ref) => {
                                    let permit = match &embedding_staleness_semaphore {
                                        Some(s) => Some(s.clone().acquire_owned().await.unwrap()),
                                        None => None,
                                    };

                                    let client = rpc_client
                                        .get_client_by_addr(sparse_ref.middleware_addr.as_str());
                                    let result = client
                                        .forward_batch_id(&(sparse_ref.clone(), is_training))
                                        .await;
                                    (result, sparse_ref.middleware_addr.clone(), permit)
                                }
                                EmbeddingTensor::Null => {
                                    panic!("current sparse data not support null data",)
                                }
                            };

                        if let Err(err) = embeddings_rpc_result {
                            tracing::error!(
                                "forward data failed {:?}, middleware: {:?}, wait embedding server recovery service",
                                err,
                                middleware_addr
                            );
                            rpc_client.wait_for_serving().await.unwrap();
                            continue;
                        }
                        let embedding_batch = embeddings_rpc_result.unwrap();

                        tracing::debug!("forward done, got embeddings");
                        if let Ok(m) = MetricsHolder::get() {
                            m.forward_client_time_cost
                                .observe(start_time.elapsed().as_secs_f64());
                        }
                        match embedding_batch {
                            Ok(embedding) => {
                                let sparse_ref = match embedding.backward_ref_id {
                                    Some(backward_ref_id) => SparseBatchRemoteReference {
                                        middleware_addr,
                                        ref_id: backward_ref_id,
                                        batcher_idx: 0,
                                    },
                                    None => SparseBatchRemoteReference::default(),
                                };
                                batch.sparse_data =
                                    EmbeddingTensor::SparseBatchRemoteReference(sparse_ref);
                                if let Err(e) = channel_s
                                    .send_async((batch, embedding, embedding_staleness_permit))
                                    .await
                                {
                                    tracing::debug!(
                                        "failed to send data to forwarded_channel_s {:?}",
                                        e
                                    );
                                }
                            }
                            Err(MiddlewareServerError::EmbeddingServerError(_))
                            | Err(MiddlewareServerError::RpcError(_)) => {
                                match rpc_client.wait_for_serving().await {
                                    Ok(_) => {
                                        tracing::debug!("wait for serving success");
                                    }
                                    Err(err) => {
                                        tracing::error!(
                                            "wait for serving failed, err msg: {:?}",
                                            err
                                        );
                                    }
                                }
                            }
                            _ => {
                                tracing::error!("forward error: {:?}", embedding_batch);
                                if let Ok(m) = MetricsHolder::get() {
                                    m.forward_error.inc();
                                }
                                tracing::error!(
                                    message = "forward with id failed, continue...",
                                    error = tracing::field::debug(&embedding_batch),
                                );
                            }
                        }
                    }
                }
            });

            self.tokio_handles.push(handle);
        }
    }
}

pub fn forward_directly(batch: PersiaBatchData, device_id: i32) -> PyResult<PythonTrainBatch> {
    set_device(device_id);

    let rpc_client = PersiaCommonContext::get().rpc_client.clone();
    let async_runtime = PersiaCommonContext::get().async_runtime.clone();

    let dense: Vec<AsyncTensorOnCuda> = batch
        .dense_data
        .into_iter()
        .map(|d| cuda_dense_tensor_h2d(d).expect("cannot move dense to gpu"))
        .collect();

    let embeddings = match &batch.sparse_data {
        EmbeddingTensor::SparseBatch(sparse_batch) => {
            let _guard = async_runtime.enter();
            let (_middleware_addr, client) = rpc_client.get_random_client_with_addr();
            let embeddings: EmbeddingBatch = async_runtime
                .block_on(client.forward_batched_direct(sparse_batch))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let embeddings: Vec<AsyncEmbeddingOnCuda> = embeddings
                .batches
                .into_iter()
                .map(|feature_embedding_batch| embedding2cuda_tensor(feature_embedding_batch))
                .collect();

            embeddings
        }
        _ => Vec::new(),
    };

    let target = batch
        .target_data
        .into_iter()
        .map(|t| cuda_dense_tensor_h2d(t).expect("cannot move dense to gpu"))
        .collect();

    let infer_batch = PersiaTrainingBatchOnGpu {
        dense,
        embeddings,
        target,
        meta_data: batch.meta_data,
        middleware_server_addr: String::new(),
        forward_id: 0,
        embedding_staleness_permit: None,
    };

    let infer_batch = PythonTrainBatch { inner: infer_batch };

    Ok(infer_batch)
}

#[pyclass]
pub struct PyForward {
    inner: Forward,
}

#[pymethods]
impl PyForward {
    #[new]
    fn new(
        forward_buffer_size: usize,
        is_training: bool,
        reproducible: bool,
        embedding_staleness: Option<usize>,
    ) -> PyResult<PyForward> {
        Ok(PyForward {
            inner: Forward::new(
                forward_buffer_size,
                is_training,
                reproducible,
                embedding_staleness,
            ),
        })
    }

    fn launch(&mut self, device_id: i32, num_workers: usize) -> PyResult<()> {
        self.inner.launch(device_id, num_workers)?;
        Ok(())
    }

    fn shutdown(&mut self) -> PyResult<()> {
        self.inner.shutdown()
    }

    pub fn get_batch(&self, timeout_ms: u64, py: Python) -> PyResult<PythonTrainBatch> {
        let start_time = std::time::Instant::now();
        let receiver = self.inner.gpu_forwarded_channel_r.clone();
        let replica_info = PersiaReplicaInfo::get().expect("not in persia context");
        let rank_id = replica_info.replica_index;

        py.allow_threads(move || {
            let batch = match receiver.try_recv() {
                Ok(x) => Ok(PythonTrainBatch { inner: x }),
                Err(_) => {
                    tracing::warn!("local forwarded queue empty for rank {}!", rank_id);
                    let result = receiver.recv_timeout(Duration::from_millis(timeout_ms));
                    if let Ok(batch) = result {
                        Ok(PythonTrainBatch { inner: batch })
                    } else {
                        Err(pyo3::exceptions::PyTimeoutError::new_err(
                            "get train batch timed out",
                        ))
                    }
                }
            };

            let elapsed = start_time.elapsed().as_millis();
            if elapsed > 1 {
                tracing::warn!(
                    message = "get_train_batch takes more than 1 milli seconds",
                    took_time = tracing::field::debug(&elapsed),
                    rank_id = rank_id,
                );
                if let Ok(m) = MetricsHolder::get() {
                    m.long_get_train_batch_time_cost
                        .observe(start_time.elapsed().as_secs_f64());
                }
            }

            return batch;
        })
    }

    fn set_input_channel(&mut self, receiver: &PyPersiaBatchDataReceiver) -> PyResult<()> {
        if self.inner.input_channel.is_none() {
            self.inner.input_channel = Some(receiver.inner.clone());
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("do not set input channel again"))
        }
    }
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "forward")?;
    module.add_class::<PyForward>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
