use crate::PersiaCommonContext;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use persia_common::optim::{AdagradConfig, AdamConfig, NaiveSGDConfig, OptimizerConfig};

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
        vectorwise_shared: Option<bool>,
    ) -> () {
        let config = AdagradConfig {
            lr,
            wd,
            g_square_momentum,
            initialization,
            eps,
            vectorwise_shared: vectorwise_shared.unwrap_or(false),
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

    pub fn apply(&self) -> PyResult<()> {
        let context = PersiaCommonContext::get();
        context
            .register_optimizer(self)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "optim")?;
    module.add_class::<PyOptimizerBase>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
