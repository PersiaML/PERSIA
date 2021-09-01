use crate::PersiaError;

use std::sync::Arc;
use std::time::Duration;

use persia_libs::{
    anyhow::Result, hashbrown::HashMap, parking_lot::RwLock, rand, tokio::runtime::Runtime, tracing,
};

use persia_embedding_server::middleware_service::MiddlewareServerClient;
use persia_model_manager::PersiaPersistenceStatus;

pub struct PersiaRpcClient {
    pub clients: RwLock<HashMap<String, Arc<MiddlewareServerClient>>>,
    pub middleware_addrs: RwLock<Vec<String>>,
    pub async_runtime: Arc<Runtime>,
}

impl PersiaRpcClient {
    pub fn new(async_runtime: Arc<Runtime>) -> Self {
        Self {
            clients: RwLock::new(HashMap::new()),
            middleware_addrs: RwLock::new(vec![]),
            async_runtime,
        }
    }

    pub fn get_random_client_with_addr(&self) -> (String, Arc<MiddlewareServerClient>) {
        let middleware_addrs = self.middleware_addrs.read();
        let addr = middleware_addrs[rand::random::<usize>() % middleware_addrs.len()].as_str();
        let client = self.get_client_by_addr(addr);
        (addr.to_string(), client)
    }

    pub fn get_random_client(&self) -> Arc<MiddlewareServerClient> {
        return self.get_random_client_with_addr().1;
    }

    pub fn get_client_by_addr(&self, middleware_addr: &str) -> Arc<MiddlewareServerClient> {
        if self.clients.read().contains_key(middleware_addr) {
            self.clients.read().get(middleware_addr).unwrap().clone()
        } else {
            let _guard = self.async_runtime.enter();
            let rpc_client = persia_rpc::RpcClient::new(middleware_addr).unwrap();
            let client = Arc::new(MiddlewareServerClient::new(rpc_client));

            self.clients
                .write()
                .insert(middleware_addr.to_string(), client.clone());

            self.middleware_addrs
                .write()
                .push(middleware_addr.to_string());
            tracing::info!("created client for middleware {}", middleware_addr);
            client
        }
    }

    // TODO(zhuxuefeng): move to nats
    pub fn get_embedding_size(&self) -> Result<Vec<usize>, PersiaError> {
        let runtime = self.async_runtime.clone();
        let _guard = runtime.enter();
        let res = runtime.block_on(self.get_random_client().get_embedding_size(&()))??;
        Ok(res)
    }

    // TODO(zhuxuefeng): move to nats
    pub fn clear_embeddings(&self) -> Result<(), PersiaError> {
        let runtime = self.async_runtime.clone();
        let _guard = runtime.enter();
        runtime.block_on(self.get_random_client().clear_embeddings(&()))??;
        Ok(())
    }

    // TODO(zhuxuefeng): move to nats
    pub fn dump(&self, dst_dir: String) -> Result<(), PersiaError> {
        let runtime = self.async_runtime.clone();
        let _guard = runtime.enter();
        runtime.block_on(
            self.clients
                .read()
                .iter()
                .next()
                .expect("clients not initialized")
                .1
                .dump(&dst_dir),
        )??;
        Ok(())
    }

    // TODO(zhuxuefeng): move to nats
    pub fn load(&self, src_dir: String) -> Result<(), PersiaError> {
        let runtime = self.async_runtime.clone();
        let _guard = runtime.enter();
        runtime.block_on(
            self.clients
                .read()
                .iter()
                .next()
                .expect("clients not initialized")
                .1
                .load(&src_dir),
        )??;
        Ok(())
    }

    // TODO(zhuxuefeng): move to nats
    pub fn wait_for_serving(&self) -> Result<(), PersiaError> {
        let runtime = self.async_runtime.clone();
        let _guard = runtime.enter();
        let client = self
            .clients
            .read()
            .iter()
            .next()
            .expect("clients not initialized")
            .1
            .clone();

        loop {
            if let Ok(ready) = runtime.block_on(client.ready_for_serving(&())) {
                if ready {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_secs(5));
                let status: Vec<PersiaPersistenceStatus> =
                    runtime.block_on(client.model_manager_status(&())).unwrap();

                match self.process_status(status) {
                    Ok(_) => {}
                    Err(err_msg) => {
                        return Err(PersiaError::ServerStatusError(err_msg));
                    }
                }
            } else {
                tracing::warn!("failed to get sparse model status, retry later");
            }
        }
    }

    pub fn shutdown(&self) -> Result<(), PersiaError> {
        let runtime = self.async_runtime.clone();
        let _guard = runtime.enter();

        let client = self.get_random_client();

        match runtime.block_on(client.shutdown_server(&())) {
            Ok(response) => match response {
                Ok(_) => {
                    let clients = self.clients.read();
                    let mut futs = clients
                        .iter()
                        .map(|client| runtime.block_on(client.1.shutdown(&())));

                    let middleware_shutdown_status = futs.all(|x| x.is_ok());
                    if middleware_shutdown_status {
                        Ok(())
                    } else {
                        Err(PersiaError::ShutdownError(String::from(
                            "shutdown middleware failed",
                        )))
                    }
                }
                Err(err) => {
                    tracing::error!("shutdown server failed, Rpc error: {:?}", err);
                    Err(PersiaError::ShutdownError(err.to_string()))
                }
            },
            Err(err) => {
                tracing::error!("shutdown server failed, Rpc error: {:?}", err);
                Err(PersiaError::ShutdownError(err.to_string()))
            }
        }
    }

    // TODO(zhuxuefeng): move to nats
    pub fn wait_for_emb_dumping(&self) -> Result<(), PersiaError> {
        let runtime = self.async_runtime.clone();
        let _guard = runtime.enter();

        let client = self
            .clients
            .read()
            .iter()
            .next()
            .expect("clients not initialized")
            .1
            .clone();

        loop {
            std::thread::sleep(Duration::from_secs(5));
            let status: Result<Vec<PersiaPersistenceStatus>, _> =
                runtime.block_on(client.model_manager_status(&()));
            if let Ok(status) = status {
                if status.iter().any(|s| match s {
                    PersiaPersistenceStatus::Loading(_) => true,
                    _ => false,
                }) {
                    let err_msg = String::from("emb status is loading but waiting for dump.");
                    return Err(PersiaError::ServerStatusError(err_msg));
                }
                let num_total = status.len();
                match self.process_status(status) {
                    Ok(num_compeleted) => {
                        if num_compeleted == num_total {
                            return Ok(());
                        }
                    }
                    Err(err_msg) => {
                        return Err(PersiaError::ServerStatusError(err_msg));
                    }
                }
            } else {
                tracing::warn!("failed to get sparse model status, retry later");
            }
        }
    }

    fn process_status(&self, status: Vec<PersiaPersistenceStatus>) -> Result<usize, String> {
        let mut num_compeleted: usize = 0;
        let mut errors = Vec::new();
        status
            .into_iter()
            .enumerate()
            .for_each(|(replica_index, s)| match s {
                PersiaPersistenceStatus::Failed(e) => {
                    let err_msg = format!(
                        "emb dump FAILED for server {}, due to {}.",
                        replica_index, e
                    );
                    errors.push(err_msg);
                }
                PersiaPersistenceStatus::Loading(p) => {
                    tracing::info!(
                        "loading emb for server {}, pregress: {:?}%",
                        replica_index,
                        p * 100.0
                    );
                }
                PersiaPersistenceStatus::Idle => {
                    num_compeleted = num_compeleted + 1;
                }
                PersiaPersistenceStatus::Dumping(p) => {
                    tracing::info!(
                        "dumping emb for server {}, pregress: {:?}%",
                        replica_index,
                        p * 100.0
                    );
                }
            });
        if errors.len() > 0 {
            Err(errors.join(", "))
        } else {
            Ok(num_compeleted)
        }
    }
}
