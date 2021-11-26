use crate::metrics::MetricsHolder;
use crate::PersiaCommonContextImpl;

use core::slice;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::JoinHandle;

use persia_libs::{
    flume, half, ndarray,
    tokio::{self, sync::OwnedSemaphorePermit, task::JoinHandle as TokioJoinHandle},
    tracing,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use persia_common::grad::{
    EmbeddingGradientBatch, FeatureEmbeddingGradientBatch, Gradients,
    SkippableFeatureEmbeddingGradientBatch, SkippedGradientBatch,
};

#[derive(Debug)]
pub struct SingleSlotGradient {
    pub slot_name: String,
    pub data_ptr: u64,
    pub shape: [usize; 2],
    pub is_f16_gradient: bool,
    pub scale_factor: f32,
}

#[derive(Debug)]
pub struct SingleSlotSkippedGradient {
    pub slot_name: String,
}

#[derive(Debug)]
pub enum SkippableSingleSlotGradient {
    Skip(SingleSlotSkippedGradient),
    Gradient(SingleSlotGradient),
}

#[derive(Debug)]
pub struct GradientBatchImpl {
    pub gradients: Vec<SkippableSingleSlotGradient>,
    pub backward_ref_id: u64,
    pub embedding_worker_addr: String,
    pub embedding_staleness_permit: Arc<Option<OwnedSemaphorePermit>>,
}

#[pyclass]
pub struct GradientBatch {
    pub inner: Option<GradientBatchImpl>,
}

impl GradientBatch {
    pub fn new(
        backward_ref_id: u64,
        embedding_worker_addr: &str,
        embedding_staleness_permit: Option<OwnedSemaphorePermit>,
    ) -> Self {
        Self {
            inner: Some(GradientBatchImpl {
                gradients: vec![],
                backward_ref_id,
                embedding_worker_addr: embedding_worker_addr.to_string(),
                embedding_staleness_permit: Arc::new(embedding_staleness_permit),
            }),
        }
    }
}

#[pymethods]
impl GradientBatch {
    pub fn add_skipped_gradient(&mut self, slot_name: String) {
        self.inner
            .as_mut()
            .unwrap()
            .gradients
            .push(SkippableSingleSlotGradient::Skip(
                SingleSlotSkippedGradient { slot_name },
            ));
    }

    pub fn add_gradient(
        &mut self,
        slot_name: String,
        data_ptr: u64,
        shape: [usize; 2],
        is_f16_gradient: bool,
        scale_factor: f32,
    ) {
        self.inner
            .as_mut()
            .unwrap()
            .gradients
            .push(SkippableSingleSlotGradient::Gradient(SingleSlotGradient {
                slot_name,
                data_ptr,
                shape,
                is_f16_gradient,
                scale_factor,
            }));
    }
}

struct EmbeddingBackwardPacket {
    pub backward_ref_id: u64,
    pub embedding_worker_addr: String,
    pub embedding_staleness_permit: Option<OwnedSemaphorePermit>,
    pub embedding_gradient_batch: EmbeddingGradientBatch,
}

struct BackwardImpl {
    pub backward_channel_s: flume::Sender<GradientBatchImpl>,
    pub backward_channel_r: flume::Receiver<GradientBatchImpl>,
    pub cpu_backward_channel_s: flume::Sender<EmbeddingBackwardPacket>,
    pub cpu_backward_channel_r: flume::Receiver<EmbeddingBackwardPacket>,
    pub launch: bool,
    pub std_handles: Vec<JoinHandle<()>>,
    pub tokio_handles: Vec<TokioJoinHandle<()>>,
    pub running: Arc<AtomicBool>,
}

fn ptr2vec<T: Clone>(ptr: *mut std::os::raw::c_void, element_num: usize) -> Vec<T> {
    unsafe {
        // NOTE: Current vector construct from ffi C++ pointer the ownership is at the C++ side
        // use vector from_raw_part will occur double-free issue.
        // TODO: Optimization points try to construct vector with zero copy
        slice::from_raw_parts(ptr as *mut T, element_num).to_vec()
    }
}

fn convert_data_ptr2gradient(
    ptr: *mut std::os::raw::c_void,
    shape: [usize; 2],
    num_elements: usize,
    is_f16: bool,
) -> Gradients {
    if is_f16 {
        Gradients::F16(
            ndarray::Array2::from_shape_vec(shape, ptr2vec::<half::f16>(ptr, num_elements))
                .unwrap(),
        )
    } else {
        Gradients::F32(
            ndarray::Array2::from_shape_vec(shape, ptr2vec::<f32>(ptr, num_elements)).unwrap(),
        )
    }
}

#[cfg(feature = "cuda")]
#[inline]
fn copy_gradients(
    x: &SingleSlotGradient,
    num_bytes: usize,
    num_elements: usize,
    device_id: Arc<Option<i32>>,
) -> Gradients {
    use crate::cuda::pinned_memory_pool::PINNED_MEMORY_POOL;
    use crate::cuda::utils::cuda_d2h;

    // Discriminate the tensor device type by device_id
    if device_id.as_ref().is_some() {
        let host_ptr = PINNED_MEMORY_POOL.allocate(num_bytes);
        let event = cuda_d2h(
            num_bytes,
            x.data_ptr as *mut std::os::raw::c_void,
            host_ptr.inner,
        )
        .expect("cannot move tensor from device to host");

        event.synchronize();

        convert_data_ptr2gradient(host_ptr.inner, x.shape, num_elements, x.is_f16_gradient)
    } else {
        convert_data_ptr2gradient(
            x.data_ptr as *mut std::os::raw::c_void,
            x.shape,
            num_elements,
            x.is_f16_gradient,
        )
    }
}

#[cfg(not(feature = "cuda"))]
#[inline]
fn copy_gradients(
    x: &SingleSlotGradient,
    _num_bytes: usize,
    num_elements: usize,
    _device_id: Arc<Option<i32>>,
) -> Gradients {
    convert_data_ptr2gradient(
        x.data_ptr as *mut std::os::raw::c_void,
        x.shape,
        num_elements,
        x.is_f16_gradient,
    )
}

impl BackwardImpl {
    fn new(queue_size: usize) -> Self {
        let (backward_channel_s, backward_channel_r) = flume::bounded(queue_size);
        let (cpu_backward_channel_s, cpu_backward_channel_r) = flume::bounded(queue_size);
        Self {
            backward_channel_s,
            backward_channel_r,
            cpu_backward_channel_s,
            cpu_backward_channel_r,
            launch: false,
            std_handles: Vec::new(),
            tokio_handles: Vec::new(),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    fn launch(&mut self, num_backward_worker: usize) {
        if !self.launch {
            self.running.store(true, Ordering::Relaxed);
            self.spawn_backward_to_cpu_worker();
            self.spawn_backward_worker(num_backward_worker);
            self.launch = true;
        }
    }

    fn shutdown(&mut self) {
        tracing::info!("exiting persia backward context");
        self.running.store(false, Ordering::Relaxed);
    }

    fn spawn_backward_to_cpu_worker(&mut self) {
        let channel_r = self.backward_channel_r.clone();
        let channel_s = self.cpu_backward_channel_s.clone();

        let running = self.running.clone();
        let device_id = PersiaCommonContextImpl::get().device_id.clone();
        let handler = std::thread::spawn(move || {
            #[cfg(feature = "cuda")]
            if let Some(device_id) = device_id.as_ref() {
                use crate::cuda::set_device;
                set_device(*device_id as i32);
            }
            loop {
                if !running.load(Ordering::Acquire) {
                    break;
                }
                let start_time = std::time::Instant::now();
                if let Ok(gradients) = channel_r.recv() {
                    tracing::debug!("get gradient batch time cost {:?}", start_time.elapsed());
                    let grads = gradients.gradients.into_iter().map(|single_slot_grad| {
                        match single_slot_grad {
                            SkippableSingleSlotGradient::Skip(x) => {
                                SkippableFeatureEmbeddingGradientBatch::Skipped(
                                    SkippedGradientBatch {
                                        feature_name: x.slot_name,
                                    },
                                )
                            }
                            SkippableSingleSlotGradient::Gradient(x) => {
                                let num_elements = x.shape[0] * x.shape[1];
                                let num_bytes = if x.is_f16_gradient {
                                    num_elements * std::mem::size_of::<half::f16>()
                                } else {
                                    num_elements * std::mem::size_of::<f32>()
                                };

                                let gradients =
                                    copy_gradients(&x, num_bytes, num_elements, device_id.clone());

                                SkippableFeatureEmbeddingGradientBatch::GradientBatch(
                                    FeatureEmbeddingGradientBatch {
                                        feature_name: x.slot_name,
                                        gradients: gradients,
                                        scale_factor: x.scale_factor,
                                    },
                                )
                            }
                        }
                    });

                    let req = EmbeddingGradientBatch {
                        gradients: grads.collect(),
                    };
                    let embedding_staleness_permit =
                        Arc::try_unwrap(gradients.embedding_staleness_permit).unwrap();

                    if let Err(e) = channel_s.send(EmbeddingBackwardPacket {
                        backward_ref_id: gradients.backward_ref_id,
                        embedding_worker_addr: gradients.embedding_worker_addr,
                        embedding_staleness_permit,
                        embedding_gradient_batch: req,
                    }) {
                        tracing::debug!("failed to send data to cpu_backward_channel_s {:?}", e);
                    }
                }
            }
        });

        self.std_handles.push(handler);
    }

    fn spawn_backward_worker(&mut self, num_backward_worker: usize) {
        let context = PersiaCommonContextImpl::get();
        let _guard = context.async_runtime.enter();

        for _ in 0..num_backward_worker {
            let channel_r = self.cpu_backward_channel_r.clone();
            let rpc_client = context.rpc_client.clone();
            let running = self.running.clone();
            let handle = tokio::spawn(async move {
                loop {
                    if !running.load(Ordering::Acquire) {
                        break;
                    }
                    let start_time = std::time::Instant::now();
                    if let Ok(embedding_backward_packet) = channel_r.recv_async().await {
                        let backward_ref_id = embedding_backward_packet.backward_ref_id;
                        let embedding_worker_addr = embedding_backward_packet.embedding_worker_addr;
                        let embedding_staleness_permit =
                            embedding_backward_packet.embedding_staleness_permit;
                        let req = embedding_backward_packet.embedding_gradient_batch;

                        tracing::debug!("get backward packet time cost {:?}", start_time.elapsed());

                        let client = rpc_client.get_client_by_addr(embedding_worker_addr.as_str());
                        let result = client
                            .update_gradient_batched(&(backward_ref_id, req))
                            .await;

                        if result.is_err() {
                            tracing::error!("backward error {:?}", result.unwrap_err());
                        } else {
                            let result = result.unwrap();
                            if result.is_err() {
                                tracing::error!("backward error {:?}", result.unwrap_err());
                            }
                        }

                        if let Some(permit) = embedding_staleness_permit {
                            drop(permit)
                        }

                        if let Ok(m) = MetricsHolder::get() {
                            m.backward_client_time_cost_sec
                                .set(start_time.elapsed().as_secs_f64());
                        }
                    }
                }
            });
            self.tokio_handles.push(handle);
        }
    }
}

#[pyclass]
struct Backward {
    inner: BackwardImpl,
}

#[pymethods]
impl Backward {
    #[new]
    pub fn new(queue_size: usize) -> Backward {
        Backward {
            inner: BackwardImpl::new(queue_size),
        }
    }

    pub fn launch(&mut self, num_backward_worker: usize) {
        self.inner.launch(num_backward_worker);
    }

    pub fn shutdown(&mut self) {
        self.inner.shutdown()
    }

    pub fn update_id_type_feature_gradient_batched(
        &self,
        gradients: &mut GradientBatch,
    ) -> PyResult<()> {
        let start_time = std::time::Instant::now();
        if let Err(err) = self
            .inner
            .backward_channel_s
            .send(gradients.inner.take().expect("cannot find gradient batch"))
        {
            return Err(PyRuntimeError::new_err(err.to_string()));
        };

        let elapsed = start_time.elapsed().as_millis();
        if elapsed > 1 {
            tracing::warn!(
                message = "update_id_type_feature_gradient_batched takes more than 1 milli seconds",
                took_time = tracing::field::debug(&elapsed)
            );
            if let Ok(m) = MetricsHolder::get() {
                m.update_gradient_batched_time_cost_more_than_1ms_sec
                    .set(start_time.elapsed().as_secs_f64());
            }
        }
        Ok(())
    }
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "backward")?;
    module.add_class::<Backward>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
