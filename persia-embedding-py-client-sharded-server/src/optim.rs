use crate::PersiaRpcClient;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use persia_embedding_datatypes::optim::Optimizer;

#[pyclass]
pub struct PyOptimizerBase {
    inner: Option<Optimizer>,
}

impl PyOptimizerBase {
    pub fn get_inner(&self) -> Option<Optimizer> {
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
        self.inner = Some(Optimizer::new_adagrad(
            lr,
            wd,
            g_square_momentum,
            initialization,
            eps,
        ));
        // self.register_optimizer2middleware()
    }

    pub fn init_sgd(&mut self, lr: f32, wd: f32) -> () {
        self.inner = Some(Optimizer::new_sgd(lr, wd));
        // self.register_optimizer2middleware()
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
