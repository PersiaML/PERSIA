#[cfg(feature = "cuda")]
use crate::cuda::set_device;

use crate::backward::GradientBatch;
use crate::data::{EmbeddingTensor, PersiaBatchImpl};
use crate::dlpack::DLManagedTensor;
use crate::metrics::MetricsHolder;
use crate::tensor::{CPUStorage, DTypeImpl, Storage, TensorImpl};
use crate::utils::PersiaBatchDataReceiver;
use crate::PersiaCommonContextImpl;

use std::collections::BinaryHeap;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use persia_common::{EmbeddingBatch, FeatureEmbeddingBatch, IDTypeFeatureRemoteRef};

use persia_embedding_config::PersiaReplicaInfo;
use persia_embedding_server::embedding_worker_service::EmbeddingWorkerError;

use numpy::{PyArray, PyArray2};
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
pub enum EmbeddingImpl {
    Raw(RawEmbedding),
    Sum(SumEmbedding),
}

#[derive(Debug)]
pub struct SumEmbedding {
    pub tensor: TensorImpl,
}

#[derive(Debug)]
pub struct RawEmbedding {
    pub tensor: TensorImpl,
    pub index: TensorImpl,
    pub non_empty_index: TensorImpl,
    pub samples_id_num: Vec<usize>,
}

#[pyclass]
pub struct Embedding {
    inner: Option<EmbeddingImpl>,
}

#[pymethods]
impl Embedding {
    pub fn is_raw_embedding(&self) -> bool {
        match self.inner.as_ref().unwrap() {
            EmbeddingImpl::Raw(_) => true,
            EmbeddingImpl::Sum(_) => false,
        }
    }

    pub fn get_sum_embedding(&mut self) -> Tensor {
        if let EmbeddingImpl::Sum(sum_embedding) = self.inner.take().unwrap() {
            Tensor {
                inner: sum_embedding.tensor,
            }
        } else {
            panic!("AttrError: raw embedding can not convert to sum embedding")
        }
    }

    pub fn get_raw_embedding(&mut self) -> (Tensor, Tensor, Tensor, Vec<usize>) {
        if let EmbeddingImpl::Raw(raw_embedding) = self.inner.take().unwrap() {
            (
                Tensor {
                    inner: raw_embedding.tensor,
                },
                Tensor {
                    inner: raw_embedding.index,
                },
                Tensor {
                    inner: raw_embedding.non_empty_index,
                },
                raw_embedding.samples_id_num,
            )
        } else {
            panic!("AttrError: sum embedding can not convert to raw embedding")
        }
    }
}

#[pyclass]
pub struct Dtype {
    inner: DTypeImpl,
}

#[pymethods]
impl Dtype {
    #[getter]
    pub fn type_id(&self) -> u8 {
        *&self.inner as u8
    }

    #[getter]
    pub fn type_name(&self) -> String {
        self.inner.get_type_name()
    }
}

#[pyclass]
pub struct Tensor {
    inner: TensorImpl,
}

static DL_TENSOR_NAME: &'static [u8] = b"dltensor\0";
const UNKONW_TENSOR_NAME: &str = "UnkownedTensor";

#[pymethods]
impl Tensor {
    #[new]
    pub fn from_numpy(data: &PyArray2<f32>) -> Tensor {
        let shape = data.shape().to_vec();

        Tensor {
            inner: TensorImpl::new(
                Storage::CPU(CPUStorage::F32(
                    data.to_vec().expect("convert ndarray to vec failed"),
                )),
                shape,
                None,
                None,
            ),
        }
    }

    #[getter]
    pub fn get_data_ptr(&mut self) -> u64 {
        self.inner.data_ptr()
    }

    #[getter]
    pub fn get_shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    pub fn get_dtype(&self) -> Dtype {
        Dtype {
            inner: self.inner.dtype(),
        }
    }

    #[getter]
    pub fn get_name(&self) -> String {
        if let Some(name) = self.inner.name.as_ref() {
            name.to_string()
        } else {
            UNKONW_TENSOR_NAME.to_owned()
        }
    }

    #[getter]
    pub fn get_dlpack(&mut self, py: Python) -> PyResult<PyObject> {
        let dlpack = self.inner.dlpack();
        tracing::debug!(
            "dlpack struct size is {:?}",
            std::mem::size_of::<DLManagedTensor>()
        );
        let dlpack_managed_tensor = Box::new(dlpack);
        let capsule = unsafe {
            let ptr = pyo3::ffi::PyCapsule_New(
                &*dlpack_managed_tensor as *const DLManagedTensor as *mut _,
                DL_TENSOR_NAME.as_ptr() as *const c_char,
                None,
            );

            if ptr.is_null() {
                return Err(PyRuntimeError::new_err(
                    "convert dlpack pointer to  capsule object failed",
                ));
            }

            PyObject::from_owned_ptr(py, ptr)
        };
        Box::leak(dlpack_managed_tensor);
        Ok(capsule)
    }

    pub fn check_dlpack(&self, dlpack: PyObject) {
        // dlpack object can not be used after dlpack checked
        // since the object already be dropped
        let dlpack_managed_tensor = unsafe {
            let ptr = pyo3::ffi::PyCapsule_GetPointer(
                dlpack.into_ptr(),
                DL_TENSOR_NAME.as_ptr() as *const c_char,
            );
            tracing::info!("dlmanaged tensor address is {:?}", ptr);

            if ptr.is_null() {
                tracing::info!("dlpack ptr empty pointer");
            }
            Box::from_raw(ptr as *mut DLManagedTensor)
        };
        tracing::info!("dlpack manager tensor is {:?}", dlpack_managed_tensor);
    }

    #[getter]
    pub fn get_device(&self) -> String {
        self.inner.device()
    }

    pub fn numpy(&self, py: Python) -> PyResult<PyObject> {
        if let Storage::CPU(storage) = &self.inner.storage {
            // PyArray::from_array(py, arr)
            let py_obj = match storage {
                // TODO:
                // * Implement half array to avoid convert half to f32. current workaround
                //   due to rust numpy no implement half, convert the half to f32 is needed.
                //   https://github.com/PyO3/rust-numpy/issues/201
                // * Use PyArrayDyn to implement dynamic shape of ndarray to avoid reshape ndarray at
                //   python side, current corruption is due to persia-speedy is not compatible with
                //   ndarray 0.14.0
                CPUStorage::F16(_val) => panic!(
                    "float16 numpy array conversion failed, pyo3 numpy is not support float16 now"
                ),
                CPUStorage::BOOL(val) => PyArray::from_slice(py, val.as_slice()).into_py(py),
                CPUStorage::F32(val) => PyArray::from_slice(py, val.as_slice()).into_py(py),
                CPUStorage::F64(val) => PyArray::from_slice(py, val.as_slice()).into_py(py),
                CPUStorage::I8(val) => PyArray::from_slice(py, val.as_slice()).into_py(py),
                CPUStorage::I16(val) => PyArray::from_slice(py, val.as_slice()).into_py(py),
                CPUStorage::I32(val) => PyArray::from_slice(py, val.as_slice()).into_py(py),
                CPUStorage::I64(val) => PyArray::from_slice(py, val.as_slice()).into_py(py),
                CPUStorage::U8(val) => PyArray::from_slice(py, val.as_slice()).into_py(py),
                CPUStorage::U16(val) => PyArray::from_slice(py, val.as_slice()).into_py(py),
                CPUStorage::U32(val) => PyArray::from_slice(py, val.as_slice()).into_py(py),
                CPUStorage::U64(val) => PyArray::from_slice(py, val.as_slice()).into_py(py),
            };
            Ok(py_obj)
        } else {
            Err(PyRuntimeError::new_err(
                "cast gpu tensor to cpu tensor before convert the tesnor to numpy, ",
            ))
        }
    }
}

#[pyclass(dict)]
pub struct PersiaTrainingBatch {
    pub inner: PersiaTrainingBatchImpl,
}

#[pymethods]
impl PersiaTrainingBatch {
    pub fn embedding_worker_addr(&self) -> &str {
        self.inner.embedding_worker_addr.as_str()
    }

    pub fn consume_all_non_id_type_feature_tensors(&mut self) -> Vec<Tensor> {
        std::mem::replace(&mut self.inner.non_id_type_feature_tensors, vec![])
            .into_iter()
            .map(|x| Tensor { inner: x })
            .collect()
    }

    pub fn consume_all_id_type_feature_embedding_tensors(&mut self) -> Vec<Embedding> {
        std::mem::replace(&mut self.inner.id_type_feature_embedding_tensors, vec![])
            .into_iter()
            .map(|x| Embedding { inner: Some(x) })
            .collect()
    }

    pub fn consume_all_label_tensors(&mut self) -> Vec<Tensor> {
        std::mem::replace(&mut self.inner.label_tensors, vec![])
            .into_iter()
            .map(|x| Tensor { inner: x })
            .collect()
    }

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

    pub fn create_gradient_batch(&mut self) -> GradientBatch {
        GradientBatch::new(
            self.inner.ref_id,
            self.inner.embedding_worker_addr.as_str(),
            self.inner.embedding_staleness_permit.take(),
        )
    }
}

#[derive(Debug)]
pub struct PersiaTrainingBatchImpl {
    pub non_id_type_feature_tensors: Vec<TensorImpl>,
    pub id_type_feature_embedding_tensors: Vec<EmbeddingImpl>,
    pub label_tensors: Vec<TensorImpl>,
    pub meta_data: Option<Vec<u8>>,
    pub embedding_worker_addr: String,
    pub ref_id: u64,
    pub embedding_staleness_permit: Option<OwnedSemaphorePermit>,
}

impl Default for PersiaTrainingBatchImpl {
    fn default() -> Self {
        Self {
            non_id_type_feature_tensors: Vec::new(),
            id_type_feature_embedding_tensors: Vec::new(),
            label_tensors: Vec::new(),
            meta_data: None,
            embedding_worker_addr: String::new(),
            ref_id: 0,
            embedding_staleness_permit: None,
        }
    }
}

fn embedding2tensor(embedding: FeatureEmbeddingBatch, device: &Option<i32>) -> EmbeddingImpl {
    match embedding {
        FeatureEmbeddingBatch::RawEmbedding(raw_embedding) => {
            let mut non_empty_index_list = Vec::new();

            raw_embedding
                .index
                .iter()
                .enumerate()
                .for_each(|(idx, id2idx)| {
                    if *id2idx != 0 {
                        non_empty_index_list.push(idx as i64);
                    }
                });

            let embedding_shape = raw_embedding.embeddings.shape().to_vec();
            let index_len = raw_embedding.index.len();
            let feature_name = raw_embedding.feature_name.clone();
            let no_empty_index_list_len = std::cmp::max(non_empty_index_list.len(), 1);

            let tensor = TensorImpl::new(
                Storage::CPU(CPUStorage::F16(raw_embedding.embeddings.into_raw_vec())),
                embedding_shape,
                Some(feature_name.clone()),
                None,
            );

            let index = TensorImpl::new(
                Storage::CPU(CPUStorage::I64(raw_embedding.index)),
                vec![index_len],
                Some(format!("{}_index", &feature_name)),
                None,
            );

            let non_empty_index = TensorImpl::new(
                Storage::CPU(CPUStorage::I64(non_empty_index_list)),
                vec![no_empty_index_list_len],
                Some(format!("{}_non_empty_index", &feature_name)),
                None,
            );

            EmbeddingImpl::Raw(RawEmbedding {
                tensor: tensor.to(&device),
                index: index.to(&device),
                non_empty_index: non_empty_index.to(&device),
                samples_id_num: raw_embedding.sample_id_num,
            })
        }
        FeatureEmbeddingBatch::SumEmbedding(sum_embedding) => {
            let embedding_shape = sum_embedding.embeddings.shape().to_vec();
            let tensor = TensorImpl::new(
                Storage::CPU(CPUStorage::F16(sum_embedding.embeddings.into_raw_vec())),
                embedding_shape,
                Some(sum_embedding.feature_name),
                None,
            );
            EmbeddingImpl::Sum(SumEmbedding {
                tensor: tensor.to(&device),
            })
        }
    }
}

struct PerisaDataOrderManager {
    pub world_size: usize,
    pub expect_batch_id: usize,
    pub data_buffer: BinaryHeap<PersiaBatchImpl>,
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
        channel_r: flume::Receiver<PersiaBatchImpl>,
        channel_s: flume::Sender<PersiaBatchImpl>,
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

    fn pop_from_buffer(&mut self) -> Option<PersiaBatchImpl> {
        if let Some(poped) = self.data_buffer.pop() {
            self.latest_pop = Instant::now();
            self.expect_batch_id = poped.batch_id.unwrap_or(self.expect_batch_id) + self.world_size;
            Some(poped)
        } else {
            None
        }
    }
}

struct ForwardImpl {
    pub input_channel: Option<flume::Receiver<PersiaBatchImpl>>,
    pub reorder_buffer_channel_s: Option<flume::Sender<PersiaBatchImpl>>,
    pub reorder_buffer_channel_r: Option<flume::Receiver<PersiaBatchImpl>>,
    pub forwarded_channel_s: flume::Sender<(
        PersiaBatchImpl,
        EmbeddingBatch,
        Option<OwnedSemaphorePermit>,
    )>,
    pub forwarded_channel_r: flume::Receiver<(
        PersiaBatchImpl,
        EmbeddingBatch,
        Option<OwnedSemaphorePermit>,
    )>,
    pub gpu_forwarded_channel_s: flume::Sender<PersiaTrainingBatchImpl>,
    pub gpu_forwarded_channel_r: flume::Receiver<PersiaTrainingBatchImpl>,
    pub launch: bool,
    pub embedding_staleness_semaphore: Option<Arc<Semaphore>>,
    pub std_handles: Vec<std::thread::JoinHandle<()>>,
    pub tokio_handles: Vec<TokioJoinHandle<()>>,
    pub running: Arc<AtomicBool>,
}

impl ForwardImpl {
    fn new(
        forward_buffer_size: usize,
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
                    self.spawn_postprocess_worker();
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

    fn spawn_postprocess_worker(&mut self) {
        let channel_r = self.forwarded_channel_r.clone();
        let channel_s = self.gpu_forwarded_channel_s.clone();

        let running = self.running.clone();
        let common_ctx = PersiaCommonContextImpl::get();

        let handler = std::thread::spawn(move || {
            #[cfg(feature = "cuda")]
            {
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
                    let id_type_feature_embedding_tensors: Vec<_> = embeddings
                        .batches
                        .into_iter()
                        .map(|feature_embedding_batch| {
                            embedding2tensor(feature_embedding_batch, common_ctx.device_id.as_ref())
                        })
                        .collect();

                    let non_id_type_feature_tensors: Vec<TensorImpl> = batch
                        .non_id_type_features
                        .into_iter()
                        .map(|d| d.to(common_ctx.device_id.as_ref()))
                        .collect();

                    let label_tensors: Vec<TensorImpl> = batch
                        .labels
                        .into_iter()
                        .map(|t| t.to(common_ctx.device_id.as_ref()))
                        .collect();

                    let (embedding_worker_addr, ref_id) =
                        batch.id_type_features.get_remote_ref_info();
                    let training_batch = PersiaTrainingBatchImpl {
                        non_id_type_feature_tensors,
                        id_type_feature_embedding_tensors,
                        label_tensors,
                        meta_data: batch.meta_data,
                        embedding_worker_addr: embedding_worker_addr.to_string(),
                        ref_id,
                        embedding_staleness_permit,
                    };
                    if let Err(e) = channel_s.send(training_batch) {
                        tracing::debug!("failed to send data to gpu_forwarded_channel_s {:?}", e);
                    }

                    if let Ok(m) = MetricsHolder::get() {
                        m.forward_client_to_gpu_time_cost_sec
                            .set(start_time.elapsed().as_secs_f64());
                    }
                }
            }
        });

        self.std_handles.push(handler);
    }

    fn spawn_forward_worker(&mut self, num_workers: usize) {
        let context = PersiaCommonContextImpl::get();
        let _guard = context.async_runtime.enter();

        for _ in 0..num_workers {
            let channel_s = self.forwarded_channel_s.clone();
            let channel_r = if let Some(reorder_channel) = self.reorder_buffer_channel_r.as_ref() {
                reorder_channel.clone()
            } else {
                self.input_channel.as_ref().unwrap().clone()
            };

            let rpc_client = context.rpc_client.clone();

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
                        let (
                            embeddings_rpc_result,
                            embedding_worker_addr,
                            embedding_staleness_permit,
                        ) = match batch.id_type_features {
                            EmbeddingTensor::IDTypeFeature(mut id_type_features) => {
                                let (embedding_worker_addr, client) =
                                    rpc_client.get_random_client_with_addr();

                                let result = client.forward_batched_direct(&id_type_features).await;

                                (result, embedding_worker_addr, None)
                            }
                            EmbeddingTensor::IDTypeFeatureRemoteRef(id_type_features_ref) => {
                                let permit = match &embedding_staleness_semaphore {
                                    Some(s) => Some(s.clone().acquire_owned().await.unwrap()),
                                    None => None,
                                };

                                let client = rpc_client.get_client_by_addr(
                                    id_type_features_ref.embedding_worker_addr.as_str(),
                                );
                                let result =
                                    client.forward_batch_id(&id_type_features_ref.clone()).await;
                                (
                                    result,
                                    id_type_features_ref.embedding_worker_addr.clone(),
                                    permit,
                                )
                            }
                            EmbeddingTensor::Null => {
                                panic!("current id type feature not support null data",)
                            }
                        };

                        if let Err(err) = embeddings_rpc_result {
                            tracing::error!(
                                "forward data failed {:?}, embedding worker: {:?}, wait embedding parameter server recovery service",
                                err,
                                embedding_worker_addr
                            );
                            rpc_client.wait_for_serving().await.unwrap();
                            continue;
                        }
                        let embedding_batch = embeddings_rpc_result.unwrap();

                        tracing::debug!("forward done, got embeddings");
                        if let Ok(m) = MetricsHolder::get() {
                            m.forward_client_time_cost_sec
                                .set(start_time.elapsed().as_secs_f64());
                        }
                        match embedding_batch {
                            Ok(embedding) => {
                                let id_type_feature_remote_ref = match embedding.backward_ref_id {
                                    Some(backward_ref_id) => IDTypeFeatureRemoteRef {
                                        embedding_worker_addr,
                                        ref_id: backward_ref_id,
                                        batcher_idx: 0,
                                    },
                                    None => IDTypeFeatureRemoteRef::default(), // batch without gradient backward
                                };
                                batch.id_type_features = EmbeddingTensor::IDTypeFeatureRemoteRef(
                                    id_type_feature_remote_ref,
                                );

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
                            Err(EmbeddingWorkerError::EmbeddingParameterServerError(_))
                            | Err(EmbeddingWorkerError::RpcError(_)) => {
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

pub fn forward_directly(
    batch: PersiaBatchImpl,
    device_id: Option<i32>,
) -> PyResult<PersiaTrainingBatch> {
    let device_id = device_id.or(PersiaCommonContextImpl::get().device_id.as_ref().clone());
    let rpc_client = PersiaCommonContextImpl::get().rpc_client.clone();
    let async_runtime = PersiaCommonContextImpl::get().async_runtime.clone();

    let non_id_type_tensors: Vec<TensorImpl> = batch
        .non_id_type_features
        .into_iter()
        .map(|d| d.to(&device_id))
        .collect();

    let embeddings = match &batch.id_type_features {
        EmbeddingTensor::IDTypeFeature(id_type_features) => {
            let _guard = async_runtime.enter();
            let (_embedding_worker_addr, client) = rpc_client.get_random_client_with_addr();
            let embeddings: EmbeddingBatch = async_runtime
                .block_on(client.forward_batched_direct(id_type_features))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let embeddings: Vec<EmbeddingImpl> = embeddings
                .batches
                .into_iter()
                .map(|feature_embedding_batch| {
                    embedding2tensor(feature_embedding_batch, &device_id)
                })
                .collect();

            embeddings
        }
        _ => Vec::new(),
    };

    let label_tensors = batch.labels.into_iter().map(|t| t.to(&device_id)).collect();

    let infer_batch = PersiaTrainingBatchImpl {
        non_id_type_feature_tensors: non_id_type_tensors,
        id_type_feature_embedding_tensors: embeddings,
        label_tensors,
        meta_data: batch.meta_data,
        ..PersiaTrainingBatchImpl::default()
    };

    let infer_batch = PersiaTrainingBatch { inner: infer_batch };

    Ok(infer_batch)
}

#[pyclass]
pub struct Forward {
    inner: ForwardImpl,
}

#[pymethods]
impl Forward {
    #[new]
    fn new(
        forward_buffer_size: usize,
        reproducible: bool,
        embedding_staleness: Option<usize>,
    ) -> PyResult<Forward> {
        Ok(Forward {
            inner: ForwardImpl::new(forward_buffer_size, reproducible, embedding_staleness),
        })
    }

    fn launch(&mut self, num_workers: usize) -> PyResult<()> {
        self.inner.launch(num_workers)?;
        Ok(())
    }

    fn shutdown(&mut self) -> PyResult<()> {
        self.inner.shutdown()
    }

    pub fn get_batch(&self, timeout_ms: u64, py: Python) -> PyResult<PersiaTrainingBatch> {
        let start_time = std::time::Instant::now();
        let receiver = self.inner.gpu_forwarded_channel_r.clone();
        let replica_info = PersiaReplicaInfo::get().expect("not in persia context");
        let rank_id = replica_info.replica_index;

        py.allow_threads(move || {
            let batch = match receiver.try_recv() {
                Ok(x) => Ok(PersiaTrainingBatch { inner: x }),
                Err(_) => {
                    tracing::warn!("local forwarded queue empty for rank {}!", rank_id);
                    let result = receiver.recv_timeout(Duration::from_millis(timeout_ms));
                    if let Ok(batch) = result {
                        Ok(PersiaTrainingBatch { inner: batch })
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
                    m.get_train_batch_time_cost_more_than_1ms_sec
                        .set(start_time.elapsed().as_secs_f64());
                }
            }

            return batch;
        })
    }

    fn set_input_channel(&mut self, receiver: &PersiaBatchDataReceiver) -> PyResult<()> {
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
    module.add_class::<Forward>()?;
    module.add_class::<Tensor>()?;
    module.add_class::<PersiaTrainingBatch>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
