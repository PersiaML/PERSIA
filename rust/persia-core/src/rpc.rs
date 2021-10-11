use crate::PersiaError;

use std::sync::Arc;
use std::time::Duration;

use persia_libs::{
    anyhow::Result, futures, indexmap::IndexMap, parking_lot::RwLock, rand, tracing,
};

use persia_embedding_server::middleware_service::MiddlewareServerClient;
use persia_model_manager::PersiaPersistenceStatus;

pub struct PersiaRpcClient {
    pub clients: RwLock<IndexMap<String, Arc<MiddlewareServerClient>>>,
}

impl PersiaRpcClient {
    pub fn new() -> Self {
        Self {
            clients: RwLock::new(IndexMap::new()),
        }
    }

    pub fn get_random_client_with_addr(&self) -> (String, Arc<MiddlewareServerClient>) {
        let clients = self.clients.read();
        let client_idx = rand::random::<usize>() % clients.len();
        let (middleware_addr, client) = clients.get_index(client_idx).unwrap();
        (middleware_addr.to_string(), client.clone())
    }

    pub fn get_random_client(&self) -> Arc<MiddlewareServerClient> {
        return self.get_random_client_with_addr().1;
    }

    pub fn get_first_client(&self) -> Arc<MiddlewareServerClient> {
        let clients = self.clients.read();
        clients
            .get_index(0)
            .expect("clients not initialized")
            .1
            .clone()
    }

    pub fn get_client_by_addr(&self, middleware_addr: &str) -> Arc<MiddlewareServerClient> {
        if self.clients.read().contains_key(middleware_addr) {
            self.clients.read().get(middleware_addr).unwrap().clone()
        } else {
            let rpc_client = persia_rpc::RpcClient::new(middleware_addr).unwrap();
            let client = Arc::new(MiddlewareServerClient::new(rpc_client));

            tracing::debug!("created client for middleware {}", middleware_addr);
            self.clients
                .write()
                .insert(middleware_addr.to_string(), client.clone());
            client
        }
    }

    // TODO(zhuxuefeng): move to nats
    pub async fn get_embedding_size(&self) -> Result<Vec<usize>, PersiaError> {
        let res = self.get_random_client().get_embedding_size(&()).await??;
        Ok(res)
    }

    // TODO(zhuxuefeng): move to nats
    pub async fn clear_embeddings(&self) -> Result<(), PersiaError> {
        self.get_random_client().clear_embeddings(&()).await??;
        Ok(())
    }

    // TODO(zhuxuefeng): move to nats
    pub async fn dump(&self, dst_dir: String) -> Result<(), PersiaError> {
        self.get_first_client().dump(&dst_dir).await??;
        Ok(())
    }

    // TODO(zhuxuefeng): move to nats
    pub async fn load(&self, src_dir: String) -> Result<(), PersiaError> {
        self.get_first_client().load(&src_dir).await??;
        Ok(())
    }

    // TODO(zhuxuefeng): move to nats
    pub async fn wait_for_serving(&self) -> Result<(), PersiaError> {
        let client = self.get_first_client().clone();

        loop {
            if let Ok(ready) = client.ready_for_serving(&()).await {
                if ready {
                    return Ok(());
                }
            } else {
                tracing::warn!("failed to get sparse model status, retry later");
                std::thread::sleep(Duration::from_secs(5));
            }
        }
    }

    pub async fn wait_for_emb_loading(&self) -> Result<(), PersiaError> {
        let client = self.get_first_client().clone();

        loop {
            if let Ok(ready) = client.ready_for_serving(&()).await {
                if ready {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_secs(5));
                let status: Vec<PersiaPersistenceStatus> =
                    client.model_manager_status(&()).await.unwrap();

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

    pub async fn shutdown(&self) -> Result<(), PersiaError> {
        let client = self.get_random_client();

        match client.shutdown_server(&()).await {
            Ok(response) => match response {
                Ok(_) => {
                    let clients = self.clients.read();
                    let futs = clients
                        .iter()
                        .map(|client| async move { client.1.shutdown(&()).await });

                    let result = futures::future::try_join_all(futs).await;

                    if result.is_ok() {
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
    pub async fn wait_for_emb_dumping(&self) -> Result<(), PersiaError> {
        let client = self.get_first_client().clone();

        loop {
            std::thread::sleep(Duration::from_secs(5));
            let status: Result<Vec<PersiaPersistenceStatus>, _> =
                client.model_manager_status(&()).await;
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
