use crate::cuda::pinned_memory_pool::PINNED_MEMORY_POOL;
use crate::cuda::set_device;
use crate::cuda::utils::cuda_d2h;

use crate::{MetricsHolder, PersiaRpcClient};

use persia_embedding_datatypes::{
    EmbeddingGradientBatch, FeatureEmbeddingGradientBatch, Gradients,
    SkippableFeatureEmbeddingGradientBatch, SkippedGradientBatch,
};
use persia_futures::{flume, tokio::sync::OwnedSemaphorePermit};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

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
pub struct GradientBatch {
    pub gradients: Vec<SkippableSingleSlotGradient>,
    pub forward_id: u64,
    pub middleware_addr: String,
    pub embedding_staleness_permit: Arc<Option<OwnedSemaphorePermit>>,
}

#[pyclass]
pub struct PythonGradientBatch {
    pub inner: Option<GradientBatch>,
}

impl PythonGradientBatch {
    pub fn new(
        forward_id: u64,
        middleware_addr: &str,
        embedding_staleness_permit: Option<OwnedSemaphorePermit>,
    ) -> Self {
        Self {
            inner: Some(GradientBatch {
                gradients: vec![],
                forward_id,
                middleware_addr: middleware_addr.to_string(),
                embedding_staleness_permit: Arc::new(embedding_staleness_permit),
            }),
        }
    }
}

#[pymethods]
impl PythonGradientBatch {
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

struct EmbeddingBackwardStub {
    pub forward_id: u64,
    pub middleware_addr: String,
    pub embedding_staleness_permit: Option<OwnedSemaphorePermit>,
    pub embedding_gradient_batch: EmbeddingGradientBatch,
}

struct Backward {
    pub backward_channel_s: flume::Sender<GradientBatch>,
    pub backward_channel_r: flume::Receiver<GradientBatch>,
    pub cpu_backward_channel_s: flume::Sender<EmbeddingBackwardStub>,
    pub cpu_backward_channel_r: flume::Receiver<EmbeddingBackwardStub>,
    pub launch: bool,
    pub threaded_workers: Vec<std::thread::JoinHandle<()>>,
}

impl Backward {
    fn new(queue_size: usize) -> Self {
        let (backward_channel_s, backward_channel_r) = flume::bounded(queue_size);
        let (cpu_backward_channel_s, cpu_backward_channel_r) = flume::bounded(queue_size);
        Self {
            backward_channel_s,
            backward_channel_r,
            cpu_backward_channel_s,
            cpu_backward_channel_r,
            launch: false,
            threaded_workers: Vec::new(),
        }
    }

    fn launch(&mut self, device_id: i32, num_backward_worker: usize) {
        if !self.launch {
            self.spawn_backward_to_cpu_worker(device_id);
            self.spawn_backward_worker(num_backward_worker);
            self.launch = true;
        }
    }

    fn spawn_backward_to_cpu_worker(&mut self, device_id: i32) {
        let channel_r = self.backward_channel_r.clone();
        let channel_s = self.cpu_backward_channel_s.clone();
        let handler = std::thread::spawn(move || {
            set_device(device_id as i32);
            loop {
                let start_time = std::time::Instant::now();
                let gradients = channel_r.recv().unwrap();
                tracing::debug!("get backward message time cost {:?}", start_time.elapsed());
                let grads =
                    gradients.gradients.into_iter().map(
                        |single_slot_grad| match single_slot_grad {
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
                                let host_ptr = PINNED_MEMORY_POOL.allocate(num_bytes);
                                let event = cuda_d2h(
                                    num_bytes,
                                    x.data_ptr as *mut std::os::raw::c_void,
                                    host_ptr.inner,
                                )
                                .expect("cannot move tensor to host");
                                // TODO: collect the event and invoke the synchronize after
                                // start d2h to improve the bandwidth
                                event.synchronize();
                                let gradients = if x.is_f16_gradient {
                                    Gradients::F16(
                                        ndarray::Array2::from_shape_vec(
                                            x.shape,
                                            host_ptr.as_slice::<half::f16>(num_elements).to_vec(),
                                        )
                                        .unwrap(),
                                    )
                                } else {
                                    Gradients::F32(
                                        ndarray::Array2::from_shape_vec(
                                            x.shape,
                                            host_ptr.as_slice::<f32>(num_elements).to_vec(),
                                        )
                                        .unwrap(),
                                    )
                                };
                                SkippableFeatureEmbeddingGradientBatch::GradientBatch(
                                    FeatureEmbeddingGradientBatch {
                                        feature_name: x.slot_name,
                                        gradients: gradients,
                                        scale_factor: x.scale_factor,
                                    },
                                )
                            }
                        },
                    );

                let req = EmbeddingGradientBatch {
                    gradients: grads.collect(),
                };
                let embedding_staleness_permit =
                    Arc::try_unwrap(gradients.embedding_staleness_permit).unwrap();

                channel_s
                    .send(EmbeddingBackwardStub {
                        forward_id: gradients.forward_id,
                        middleware_addr: gradients.middleware_addr,
                        embedding_staleness_permit,
                        embedding_gradient_batch: req,
                    })
                    .unwrap();
            }
        });
        self.threaded_workers.push(handler);
    }

    fn spawn_backward_worker(&mut self, num_backward_worker: usize) {
        let rpc_client = PersiaRpcClient::get_instance();
        let runtime = rpc_client.runtime.clone();

        for _ in 0..num_backward_worker {
            let channel_r = self.cpu_backward_channel_r.clone();
            let rpc_client = rpc_client.clone();
            let _guard = runtime.enter();
            persia_futures::tokio::spawn(async move {
                loop {
                    let start_time = std::time::Instant::now();

                    let embedding_backward_stub = channel_r.recv_async().await.unwrap();
                    let forward_id = embedding_backward_stub.forward_id;
                    let middleware_addr = embedding_backward_stub.middleware_addr;
                    let embedding_staleness_permit =
                        embedding_backward_stub.embedding_staleness_permit;
                    let req = embedding_backward_stub.embedding_gradient_batch;

                    tracing::debug!(
                        "get cpu backward message time cost {:?}",
                        start_time.elapsed()
                    );

                    let client = rpc_client.get_client_by_addr(middleware_addr.as_str());
                    let result = client.update_gradient_batched(&(forward_id, req)).await;

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
                        m.backward_client_time_cost
                            .observe(start_time.elapsed().as_secs_f64());
                    }
                }
            });
        }
    }
}

#[pyclass]
struct PyBackward {
    inner: Backward,
}

#[pymethods]
impl PyBackward {
    #[new]
    pub fn new(queue_size: usize) -> PyBackward {
        PyBackward {
            inner: Backward::new(queue_size),
        }
    }

    pub fn launch(&mut self, device_id: i32, num_backward_worker: usize) {
        self.inner.launch(device_id, num_backward_worker);
    }

    pub fn update_sparse_gradient_batched(
        &self,
        gradients: &mut PythonGradientBatch,
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
                message = "update_sparse_gradient_batched takes more than 1 milli seconds",
                took_time = tracing::field::debug(&elapsed)
            );
            if let Ok(m) = MetricsHolder::get() {
                m.long_update_gradient_batched_time_cost
                    .observe(start_time.elapsed().as_secs_f64());
            }
        }
        Ok(())
    }
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "backward")?;
    module.add_class::<PyBackward>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
