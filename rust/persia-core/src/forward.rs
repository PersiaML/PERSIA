use crate::backward::PythonGradientBatch;

use crate::metrics::MetricsHolder;
use crate::utils::PyPersiaBatchDataReceiver;
use crate::PersiaCommonContext;

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use persia_common::tensor::CPUStorage;
use persia_common::{
    tensor::{Storage, Tensor},
    EmbeddingBatch, EmbeddingTensor, FeatureEmbeddingBatch, PersiaBatchData,
};

#[cfg(feature = "cuda")]
use persia_common::cuda::set_device;

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

#[derive(Debug)]
pub enum Embedding {
    Raw(RawEmbedding),
    Sum(SumEmbedding),
}

#[derive(Debug)]
pub struct SumEmbedding {
    pub tensor: Tensor,
}

#[derive(Debug)]
pub struct RawEmbedding {
    pub tensor: Tensor,
    pub index: Tensor,
    pub non_empty_index: Tensor,
    pub samples_id_num: Vec<usize>,
}

#[pyclass]
pub struct PyEmbedding {
    inner: Option<Embedding>,
}

#[pymethods]
impl PyEmbedding {
    pub fn is_raw_embedding(&self) -> bool {
        match self.inner.as_ref().unwrap() {
            Embedding::Raw(_) => true,
            Embedding::Sum(_) => false,
        }
    }

    pub fn get_sum_embedding(&mut self) -> PyTensor {
        if let Embedding::Sum(sum_embedding) = self.inner.take().unwrap() {
            PyTensor {
                inner: sum_embedding.tensor,
                is_ready: false,
            }
        } else {
            panic!("AttrError: raw embedding can not convert to sum embedding")
        }
    }

    pub fn get_raw_embedding(&mut self) -> (PyTensor, PyTensor, PyTensor, Vec<usize>) {
        if let Embedding::Raw(raw_embedding) = self.inner.take().unwrap() {
            (
                PyTensor {
                    inner: raw_embedding.tensor,
                    is_ready: false,
                },
                PyTensor {
                    inner: raw_embedding.index,
                    is_ready: false,
                },
                PyTensor {
                    inner: raw_embedding.non_empty_index,
                    is_ready: false,
                },
                raw_embedding.samples_id_num,
            )
        } else {
            panic!("AttrError: sum embedding can not convert to raw embedding")
        }
    }
}

#[pyclass]
pub struct PyTensor {
    inner: Tensor,
    is_ready: bool,
}

#[pymethods]
impl PyTensor {
    #[cfg(feature = "cuda")]
    pub fn sync_event(&mut self) {
        if !self.is_ready {
            self.inner.storage.gpu_storage_ref().event.synchronize();
            self.is_ready = true;
        }
    }

    #[cfg(feature = "cuda")]
    pub fn data_ptr(&mut self) -> u64 {
        self.sync_event();
        self.inner.storage.gpu_storage_ref().ptr.inner as u64
    }

    #[cfg(feature = "cuda")]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[cfg(feature = "cuda")]
    pub fn num_bytes(&self) -> usize {
        self.inner.storage.gpu_storage_ref().ptr.num_bytes
    }

    pub fn numpy(&self) {}
}

#[pyclass(dict)]
pub struct PythonTrainBatch {
    pub inner: PersiaTrainingBatch,
}

#[pymethods]
impl PythonTrainBatch {
    pub fn middleware_server_addr(&self) -> &str {
        self.inner.middleware_server_addr.as_str()
    }

    pub fn consume_all_dense_features(&mut self) -> Vec<PyTensor> {
        std::mem::replace(&mut self.inner.dense, vec![])
            .into_iter()
            .map(|x| PyTensor {
                inner: x,
                is_ready: false,
            })
            .collect()
    }

    pub fn consume_all_sparse_features(&mut self) -> Vec<PyEmbedding> {
        std::mem::replace(&mut self.inner.embeddings, vec![])
            .into_iter()
            .map(|x| PyEmbedding { inner: Some(x) })
            .collect()
    }

    pub fn consume_all_targets(&mut self) -> Vec<PyTensor> {
        std::mem::replace(&mut self.inner.target, vec![])
            .into_iter()
            .map(|x| PyTensor {
                inner: x,
                is_ready: false,
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
pub struct PersiaTrainingBatch {
    pub dense: Vec<Tensor>,
    pub embeddings: Vec<Embedding>,
    pub target: Vec<Tensor>,
    pub meta_data: Option<Vec<u8>>,
    pub middleware_server_addr: String,
    pub forward_id: u64,
    pub embedding_staleness_permit: Option<OwnedSemaphorePermit>,
}

fn embedding2tensor(embedding: FeatureEmbeddingBatch) -> Embedding {
    match embedding {
        FeatureEmbeddingBatch::RawEmbedding(raw_embedding) => {
            let mut non_empty_index_list = Vec::new();

            raw_embedding
                .index
                .iter()
                .enumerate()
                .for_each(|(idx, id2idx)| {
                    if *id2idx != 0 {
                        non_empty_index_list.push(idx as u64);
                    }
                });

            Embedding::Raw(RawEmbedding {
                tensor: Tensor {
                    shape: raw_embedding.embeddings.shape().to_vec(),
                    storage: Storage::CPU(CPUStorage::from_f16(
                        raw_embedding.embeddings.into_raw_vec(),
                    )),
                    name: None,
                },
                index: Tensor {
                    shape: vec![raw_embedding.index.len()],
                    storage: Storage::CPU(CPUStorage::from_usize(raw_embedding.index)),
                    name: None,
                },
                non_empty_index: Tensor {
                    shape: vec![std::cmp::max(non_empty_index_list.len(), 1)],
                    storage: Storage::CPU(CPUStorage::from_u64(non_empty_index_list)),
                    name: None,
                },
                samples_id_num: raw_embedding.sample_id_num,
            })
        }
        FeatureEmbeddingBatch::SumEmbedding(sum_embedding) => {
            let tensor = Tensor {
                shape: sum_embedding.embeddings.shape().to_vec(),
                storage: Storage::CPU(CPUStorage::from_f16(
                    sum_embedding.embeddings.into_raw_vec(),
                )),
                name: None,
            };
            Embedding::Sum(SumEmbedding { tensor })
        }
    }
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
    pub gpu_forwarded_channel_s: flume::Sender<PersiaTrainingBatch>,
    pub gpu_forwarded_channel_r: flume::Receiver<PersiaTrainingBatch>,
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

    pub fn launch(&mut self, num_workers: usize) -> PyResult<()> {
        if !self.launch {
            match &self.input_channel {
                Some(_) => {
                    self.running.store(true, Ordering::Relaxed);
                    if self.reorder_buffer_channel_r.is_some() {
                        self.spawn_reorder_buffer_worker()?;
                    }
                    self.spawn_to_gpu_worker();
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

    fn spawn_to_gpu_worker(&mut self) {
        let channel_r = self.forwarded_channel_r.clone();
        let channel_s = self.gpu_forwarded_channel_s.clone();

        let running = self.running.clone();
        let common_ctx = PersiaCommonContext::get();

        let handler = std::thread::spawn(move || {
            let mut use_gpu = false;
            #[cfg(feature = "cuda")]
            {
                use_gpu = common_ctx.device_id.as_ref().is_some();
                if let Some(device_id) = common_ctx.device_id.as_ref() {
                    set_device(*device_id);
                }
            }

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
                        .map(|feature_embedding_batch| embedding2tensor(feature_embedding_batch))
                        .collect();

                    let dense_tensors: Vec<Tensor> = batch
                        .dense_data
                        .into_iter()
                        .map(|d| {
                            #[cfg(feature = "cuda")]
                            {
                                if use_gpu {
                                    d.cuda()
                                } else {
                                    d
                                }
                            }

                            #[cfg(not(feature = "cuda"))]
                            {
                                d
                            }
                        })
                        .collect();

                    let target_tensors: Vec<Tensor> = batch
                        .target_data
                        .into_iter()
                        .map(|t| {
                            #[cfg(feature = "cuda")]
                            {
                                if use_gpu {
                                    t.cuda()
                                } else {
                                    t
                                }
                            }

                            #[cfg(not(feature = "cuda"))]
                            {
                                t
                            }
                        })
                        .collect();

                    let (middleware_addr, forward_id) = batch.sparse_data.to_forward_id();
                    let training_batch = PersiaTrainingBatch {
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

                        let (embeddings_result, middleware_addr, embedding_staleness_permit) =
                            match &batch.sparse_data {
                                EmbeddingTensor::SparseBatch(sparse_data) => {
                                    let (middleware_addr, client) =
                                        rpc_client.get_random_client_with_addr();

                                    let result = client.forward_batched_direct(sparse_data).await;
                                    (result, middleware_addr, None)
                                }
                                EmbeddingTensor::PreForwardStub(stub) => {
                                    let permit = match &embedding_staleness_semaphore {
                                        Some(s) => Some(s.clone().acquire_owned().await.unwrap()),
                                        None => None,
                                    };

                                    let client = rpc_client
                                        .get_client_by_addr(stub.middleware_addr.as_str());
                                    let result =
                                        client.forward_batch_id(&(stub.clone(), is_training)).await;
                                    (result, stub.middleware_addr.clone(), permit)
                                }
                                EmbeddingTensor::Null => {
                                    panic!("current sparse data not support null data",)
                                }
                            };

                        if embeddings_result.is_err() {
                            tracing::error!(
                                "forward data failed {:?}, middleware: {:?}, wait embedding server recovery service",
                                embeddings_result,
                                middleware_addr
                            );
                            rpc_client.wait_for_serving().unwrap();
                            continue;
                        }
                        let embeddings = embeddings_result.unwrap();

                        tracing::debug!("forward done, got embeddings");
                        if let Ok(m) = MetricsHolder::get() {
                            m.forward_client_time_cost
                                .observe(start_time.elapsed().as_secs_f64());
                        }
                        match embeddings {
                            Ok(embeddings) => {
                                if let Err(e) = channel_s
                                    .send_async((batch, embeddings, embedding_staleness_permit))
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
                                match rpc_client.wait_for_serving() {
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
                                tracing::error!("forward error: {:?}", embeddings);
                                if let Ok(m) = MetricsHolder::get() {
                                    m.forward_error.inc();
                                }
                                tracing::error!(
                                    message = "forward with id failed, continue...",
                                    error = tracing::field::debug(&embeddings),
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

    let dense: Vec<Tensor> = batch.dense_data.into_iter().map(|d| d).collect();

    let emb_tensor = &batch.sparse_data;
    let embeddings = match emb_tensor {
        EmbeddingTensor::SparseBatch(sparse_batch) => {
            let _guard = async_runtime.enter();
            let (_middleware_addr, client) = rpc_client.get_random_client_with_addr();
            let embeddings: EmbeddingBatch = async_runtime
                .block_on(client.forward_batched_direct(sparse_batch))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let embeddings: Vec<Embedding> = embeddings
                .batches
                .into_iter()
                .map(|feature_embedding_batch| embedding2tensor(feature_embedding_batch))
                .collect();

            embeddings
        }
        _ => Vec::new(),
    };

    let target = batch.target_data.into_iter().map(|t| t).collect();

    let infer_batch = PersiaTrainingBatch {
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

    fn launch(&mut self, num_workers: usize) -> PyResult<()> {
        self.inner.launch(num_workers)?;
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
