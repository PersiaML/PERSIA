use crate::PersiaRpcClient;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use persia_embedding_datatypes::optim::{
    AdagradConfig, AdamConfig, NaiveSGDConfig, OptimizerConfig,
};

#[pyclass]
pub struct PyOptimizerBase {
    inner: Option<OptimizerConfig>,
}

impl PyOptimizerBase {
    pub fn get_inner(&self) -> Option<OptimizerConfig> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyOptimizerBase {
    #[new]
    pub fn new() -> Self {
        Self { inner: None }
    }

    pub fn init_adagrad(
        &mut self,
        lr: f32,
        wd: f32,
        g_square_momentum: f32,
        initialization: f32,
        eps: f32,
    ) -> () {
        let config = AdagradConfig {
            lr,
            wd,
            g_square_momentum,
            initialization,
            eps,
        };
        self.inner = Some(OptimizerConfig::Adagrad(config));
    }

    pub fn init_sgd(&mut self, lr: f32, wd: f32) -> () {
        let config = NaiveSGDConfig { lr, wd };
        self.inner = Some(OptimizerConfig::SGD(config));
    }

    pub fn init_adam(&mut self, lr: f32, betas: (f32, f32), eps: f32) -> () {
        let config = AdamConfig {
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps,
        };
        self.inner = Some(OptimizerConfig::Adam(config));
    }

    fn register_optimizer2middleware(&self) -> PyResult<()> {
        let rpc_client = PersiaRpcClient::get_instance();
        let runtime = rpc_client.runtime.clone();
        let _guard = runtime.enter();

        runtime
            .block_on(
                rpc_client
                    .get_random_client()
                    .register_optimizer(&self.inner.as_ref().unwrap()),
            )
            .unwrap()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "optim")?;
    module.add_class::<PyOptimizerBase>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
