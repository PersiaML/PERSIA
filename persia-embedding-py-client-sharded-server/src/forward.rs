use crate::backward::PythonGradientBatch;
use crate::cuda::utils::{cuda_dense_tensor_h2d, embedding2cuda_tensor};
use crate::cuda::{set_device, AsyncEmbeddingOnCuda, AsyncTensorOnCuda};
use crate::data::PyPersiaBatchData;
use crate::utils::{PyPersiaBatchDataChannel, PyPersiaReplicaInfo};
use crate::{MetricsHolder, PersiaRpcClient};

use std::collections::BinaryHeap;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use persia_embedding_config::PersiaReplicaInfo;
use persia_embedding_datatypes::{EmbeddingBatch, EmbeddingTensor, PersiaBatchData};
use persia_embedding_sharded_server::sharded_middleware_service::ShardedMiddlewareError;
use persia_futures::flume;
use persia_speedy::Readable;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;

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
    pub inner: PersiaTrainingBatchShardedServerOnGpu,
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

    pub fn create_gradient_batch(&self) -> PythonGradientBatch {
        PythonGradientBatch::new(
            self.inner.forward_id,
            self.inner.middleware_server_addr.as_str(),
        )
    }
}

#[derive(Debug)]
pub struct PersiaTrainingBatchShardedServerOnGpu {
    pub dense: Vec<AsyncTensorOnCuda>,
    pub embeddings: Vec<AsyncEmbeddingOnCuda>,
    pub target: Vec<AsyncTensorOnCuda>,
    pub meta_data: Option<Vec<u8>>,
    pub middleware_server_addr: String,
    pub forward_id: u64,
}

struct PerisaDataOrderManager {
    pub world_size: usize,
    pub disorder_tolerance: f32,
    pub expect_batch_id: usize,
    pub data_buffer: BinaryHeap<PersiaBatchData>,
    pub latest_pop: Instant,
}

impl PerisaDataOrderManager {
    pub fn new(world_size: usize, rank_id: usize, disorder_tolerance: f32) -> Self {
        Self {
            world_size,
            disorder_tolerance,
            expect_batch_id: rank_id,
            data_buffer: BinaryHeap::new(),
            latest_pop: Instant::now(),
        }
    }

    pub fn spawn_reorder_worker(
        world_size: usize,
        rank_id: usize,
        disorder_tolerance: f32,
        channel_r: flume::Receiver<PersiaBatchData>,
        channel_s: flume::Sender<PersiaBatchData>,
    ) -> JoinHandle<()> {
        std::thread::spawn(move || {
            if disorder_tolerance == 1.0 {
                channel_r.iter().for_each(|batch| {
                    channel_s.send(batch).unwrap();
                });
            } else {
                let mut order_manager = Self::new(world_size, rank_id, disorder_tolerance);
                loop {
                    if let Ok(batch) = channel_r.recv_timeout(Duration::from_millis(10)) {
                        order_manager.data_buffer.push(batch);
                        while let Some(data) = order_manager.data_buffer.peek() {
                            let data_batch_id = data.batch_id.unwrap_or(usize::MIN);
                            let num_absent = (data_batch_id - order_manager.expect_batch_id) as f32
                                / order_manager.world_size as f32;
                            if data_batch_id <= order_manager.expect_batch_id
                                || num_absent
                                    < order_manager.data_buffer.len() as f32
                                        * order_manager.disorder_tolerance
                            {
                                channel_s
                                    .send(order_manager.pop_from_buffer().unwrap())
                                    .unwrap();
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
                            about the order of input batches, please set disorder_tolerance to 1.0 to get a higher training efficiency."
                        );
                        while let Some(poped) = order_manager.pop_from_buffer() {
                            channel_s.send(poped).unwrap();
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
    pub nats_channel_s: flume::Sender<PersiaBatchData>,
    pub nats_channel_r: flume::Receiver<PersiaBatchData>,
    pub reorder_buffer_channel_s: flume::Sender<PersiaBatchData>,
    pub reorder_buffer_channel_r: flume::Receiver<PersiaBatchData>,
    pub forwarded_channel_s: flume::Sender<(PersiaBatchData, EmbeddingBatch)>,
    pub forwarded_channel_r: flume::Receiver<(PersiaBatchData, EmbeddingBatch)>,
    pub gpu_forwarded_channel_s: flume::Sender<PersiaTrainingBatchShardedServerOnGpu>,
    pub gpu_forwarded_channel_r: flume::Receiver<PersiaTrainingBatchShardedServerOnGpu>,
    pub forward_only: bool,
    pub launch: bool,
    pub threaded_workers: Vec<std::thread::JoinHandle<()>>,
    pub replica_info: PersiaReplicaInfo,
}

impl Forward {
    fn new(
        forward_buffer_size: usize,
        nats_recv_buffer_size: usize,
        forward_only: bool,
        replica_info: PersiaReplicaInfo,
    ) -> Self {
        let (nats_channel_s, nats_channel_r) = flume::bounded(nats_recv_buffer_size);
        let (reorder_buffer_channel_s, reorder_buffer_channel_r) = flume::bounded(1);
        let (forwarded_channel_s, forwarded_channel_r) = flume::bounded(forward_buffer_size);
        let (gpu_forwarded_channel_s, gpu_forwarded_channel_r) =
            flume::bounded(forward_buffer_size);

        Self {
            nats_channel_s,
            nats_channel_r,
            reorder_buffer_channel_s,
            reorder_buffer_channel_r,
            forwarded_channel_r,
            forwarded_channel_s,
            gpu_forwarded_channel_r,
            gpu_forwarded_channel_s,
            forward_only,
            launch: false,
            threaded_workers: Vec::new(),
            replica_info,
        }
    }

    pub fn launch(&mut self, device_id: i32, disorder_tolerance: f32, num_workers: usize) {
        if !self.launch {
            self.spawn_reorder_buffer_worker(disorder_tolerance);
            self.spawn_to_gpu_worker(device_id);
            self.spawn_forward_worker(num_workers);
            self.launch = true;
        } else {
            tracing::warn!("forward engine already launch");
        }
    }

    fn spawn_reorder_buffer_worker(&mut self, disorder_tolerance: f32) {
        let handler = PerisaDataOrderManager::spawn_reorder_worker(
            self.replica_info.replica_size,
            self.replica_info.replica_index,
            disorder_tolerance,
            self.nats_channel_r.clone(),
            self.reorder_buffer_channel_s.clone(),
        );

        self.threaded_workers.push(handler);
    }

    fn spawn_to_gpu_worker(&mut self, device_id: i32) {
        let channel_r = self.forwarded_channel_r.clone();
        let channel_s = self.gpu_forwarded_channel_s.clone();

        let handler = std::thread::spawn(move || {
            set_device(device_id);
            loop {
                let start_time = std::time::Instant::now();
                let (batch, embeddings) = channel_r.recv().unwrap();
                tracing::debug!("get forwarded batch time cost {:?}", start_time.elapsed());
                let embeddings: Vec<_> = embeddings
                    .batches
                    .into_iter()
                    .map(|feature_embedding_batch| embedding2cuda_tensor(feature_embedding_batch))
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
                let training_batch = PersiaTrainingBatchShardedServerOnGpu {
                    dense: dense_tensors,
                    embeddings,
                    target: target_tensors,
                    meta_data: batch.meta_data,
                    middleware_server_addr: middleware_addr.to_string(),
                    forward_id: forward_id,
                };
                channel_s.send(training_batch).unwrap();

                if let Ok(m) = MetricsHolder::get() {
                    m.forward_client_to_gpu_time_cost
                        .observe(start_time.elapsed().as_secs_f64());
                }
            }
        });
        self.threaded_workers.push(handler);
    }

    fn spawn_forward_worker(&mut self, num_workers: usize) {
        let rpc_client = PersiaRpcClient::get_instance();

        for _ in 0..num_workers {
            let rpc_client = rpc_client.clone();
            let channel_s = self.forwarded_channel_s.clone();
            let channel_r = self.reorder_buffer_channel_r.clone();
            let _guard = rpc_client.runtime.enter();
            let forward_only = self.forward_only;
            persia_futures::tokio::spawn(async move {
                loop {
                    let start_time = std::time::Instant::now();
                    let batch = channel_r.recv_async().await.unwrap();
                    tracing::debug!(
                        "get deserialized message time cost {:?}",
                        start_time.elapsed()
                    );
                    let (ref server_addr, forward_id) = batch.sparse_data.to_forward_id();
                    let client = rpc_client.get_client_by_addr(server_addr);
                    let embeddings_result =
                        client.forward_batch_id(&(forward_id, forward_only)).await;
                    if embeddings_result.is_err() {
                        tracing::error!(
                            "forward failed {:?}, server: {:?}, sleeping 5s",
                            embeddings_result,
                            server_addr
                        );
                        persia_futures::tokio::time::sleep(Duration::from_secs(5)).await;
                        loop {
                            if let Ok(response) = client.ready_for_serving(&()).await {
                                if response {
                                    break;
                                }
                            }
                            persia_futures::tokio::time::sleep(Duration::from_secs(30)).await;
                        }
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
                            channel_s.send_async((batch, embeddings)).await.unwrap();
                        }
                        Err(ShardedMiddlewareError::ShardServerError(_)) => loop {
                            if let Ok(response) = client.ready_for_serving(&()).await {
                                if response {
                                    break;
                                }
                            }
                            persia_futures::tokio::time::sleep(Duration::from_secs(30)).await;
                        },
                        Err(ShardedMiddlewareError::RpcError(_)) => loop {
                            if let Ok(response) = client.ready_for_serving(&()).await {
                                if response {
                                    break;
                                }
                            }
                            persia_futures::tokio::time::sleep(Duration::from_secs(30)).await;
                        },
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
            });
        }
    }
}

pub fn forward_directly(batch: PersiaBatchData, device_id: i32) -> PyResult<PythonTrainBatch> {
    set_device(device_id);

    let dense: Vec<AsyncTensorOnCuda> = batch
        .dense_data
        .into_iter()
        .map(|d| cuda_dense_tensor_h2d(d).expect("cannot move dense to gpu"))
        .collect();

    let rpc_client = PersiaRpcClient::get_instance();

    let emb_tensor = &batch.sparse_data;
    let embeddings = match emb_tensor {
        EmbeddingTensor::SparseBatch(sparse_batch) => {
            let runtime = rpc_client.runtime.clone();
            let _guard = runtime.enter();
            let (_middleware_addr, client) = rpc_client.get_random_client_with_addr();
            let embeddings: EmbeddingBatch = runtime
                .block_on(client.forward_batched_direct(sparse_batch))
                .unwrap()
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

    let infer_batch = PersiaTrainingBatchShardedServerOnGpu {
        dense,
        embeddings,
        target: Vec::new(),
        meta_data: batch.meta_data,
        middleware_server_addr: String::new(),
        forward_id: 0,
    };

    let infer_batch = PythonTrainBatch { inner: infer_batch };

    Ok(infer_batch)
}

#[pyfunction]
pub fn forward_directly_from_data(
    batch: &mut PyPersiaBatchData,
    device_id: i32,
) -> PyResult<PythonTrainBatch> {
    let batch = std::mem::replace(&mut batch.inner, PersiaBatchData::default());
    forward_directly(batch, device_id)
}

#[pyfunction]
pub fn forward_directly_from_bytes(batch: &PyBytes, device_id: i32) -> PyResult<PythonTrainBatch> {
    let batch: PersiaBatchData = PersiaBatchData::read_from_buffer(batch.as_bytes()).unwrap();
    forward_directly(batch, device_id)
}

#[pyclass]
pub struct PyForward {
    inner: Forward,
}

impl PyForward {
    pub fn get_nats_channel_s(&self) -> flume::Sender<PersiaBatchData> {
        self.inner.nats_channel_s.clone()
    }
}

#[pymethods]
impl PyForward {
    #[new]
    fn new(
        forward_buffer_size: usize,
        nats_recv_buffer_size: usize,
        forward_only: bool,
        repilca_info: &PyPersiaReplicaInfo,
    ) -> PyResult<PyForward> {
        Ok(PyForward {
            inner: Forward::new(
                forward_buffer_size,
                nats_recv_buffer_size,
                forward_only,
                repilca_info.get_replica_info(),
            ),
        })
    }

    fn launch(&mut self, device_id: i32, disorder_tolerance: f32, num_workers: usize) {
        self.inner
            .launch(device_id, disorder_tolerance, num_workers)
    }

    pub fn get_batch(&self, timeout_ms: u64) -> PyResult<PythonTrainBatch> {
        let start_time = std::time::Instant::now();
        let receiver = self.inner.gpu_forwarded_channel_r.clone();
        let batch = match receiver.try_recv() {
            Ok(x) => Ok(PythonTrainBatch { inner: x }),
            Err(_) => {
                tracing::warn!("local forwarded queue empty!");
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
                took_time = tracing::field::debug(&elapsed)
            );
            if let Ok(m) = MetricsHolder::get() {
                m.long_get_train_batch_time_cost
                    .observe(start_time.elapsed().as_secs_f64());
            }
        }

        return batch;
    }

    fn get_input_channel(&self) -> PyPersiaBatchDataChannel {
        PyPersiaBatchDataChannel {
            sender: self.get_nats_channel_s(),
        }
    }
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "forward")?;
    module.add_class::<PyForward>()?;
    module.add_function(wrap_pyfunction!(forward_directly_from_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(forward_directly_from_data, module)?)?;
    super_module.add_submodule(module)?;
    Ok(())
}
