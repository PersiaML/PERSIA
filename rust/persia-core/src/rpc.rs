use crate::PersiaError;

use std::sync::Arc;
use std::time::Duration;

use persia_libs::{
    anyhow::Result, futures, indexmap::IndexMap, itertools::Itertools, parking_lot::RwLock, rand,
    tracing,
};

use persia_embedding_holder::emb_entry::HashMapEmbeddingEntry;
use persia_embedding_server::embedding_worker_service::EmbeddingWorkerClient;
use persia_model_manager::EmbeddingModelManagerStatus;

pub struct PersiaRpcClient {
    pub clients: RwLock<IndexMap<String, Arc<EmbeddingWorkerClient>>>,
}

impl PersiaRpcClient {
    pub fn new() -> Self {
        Self {
            clients: RwLock::new(IndexMap::new()),
        }
    }

    pub fn get_random_client_with_addr(&self) -> (String, Arc<EmbeddingWorkerClient>) {
        let clients = self.clients.read();
        let client_idx = rand::random::<usize>() % clients.len();
        let (embedding_worker_addr, client) = clients.get_index(client_idx).unwrap();
        (embedding_worker_addr.to_string(), client.clone())
    }

    pub fn get_random_client(&self) -> Arc<EmbeddingWorkerClient> {
        return self.get_random_client_with_addr().1;
    }

    pub fn get_first_client(&self) -> Arc<EmbeddingWorkerClient> {
        let clients = self.clients.read();
        clients
            .get_index(0)
            .expect("clients not initialized")
            .1
            .clone()
    }

    pub fn get_client_by_index(&self, client_index: usize) -> Arc<EmbeddingWorkerClient> {
        self.clients
            .read()
            .get_index(client_index)
            .expect("clients not initialized")
            .1
            .clone()
    }

    pub fn get_client_by_addr(&self, embedding_worker_addr: &str) -> Arc<EmbeddingWorkerClient> {
        if self.clients.read().contains_key(embedding_worker_addr) {
            self.clients
                .read()
                .get(embedding_worker_addr)
                .unwrap()
                .clone()
        } else {
            let rpc_client = persia_rpc::RpcClient::new(embedding_worker_addr).unwrap();
            let client = Arc::new(EmbeddingWorkerClient::new(rpc_client));

            tracing::debug!(
                "created client for embedding worker {}",
                embedding_worker_addr
            );
            self.clients
                .write()
                .insert(embedding_worker_addr.to_string(), client.clone());
            client
        }
    }

    pub async fn set_embedding(
        &self,
        entries: Vec<HashMapEmbeddingEntry>,
    ) -> Result<(), PersiaError> {
        let num_embedding_workers = self.clients.read().len();
        let num_entries = entries.len();

        let grouped_entries: Vec<Vec<HashMapEmbeddingEntry>> = entries
            .into_iter()
            .chunks(num_entries / num_embedding_workers)
            .into_iter()
            .map(|chunk| chunk.collect())
            .collect();

        let futs = grouped_entries
            .into_iter()
            .enumerate()
            .map(|(client_index, entries)| {
                let client = self.get_client_by_index(client_index);
                async move { client.set_embedding(&entries).await }
            });

        let results = futures::future::try_join_all(futs).await?;

        for res in results {
            res?;
        }

        Ok(())
    }

    pub async fn get_embedding_size(&self) -> Result<Vec<usize>, PersiaError> {
        let res = self.get_random_client().get_embedding_size(&()).await??;
        Ok(res)
    }

    pub async fn clear_embeddings(&self) -> Result<(), PersiaError> {
        self.get_random_client().clear_embeddings(&()).await??;
        Ok(())
    }

    pub async fn dump(&self, dst_dir: String) -> Result<(), PersiaError> {
        self.get_first_client().dump(&dst_dir).await??;
        Ok(())
    }

    pub async fn load(&self, src_dir: String) -> Result<(), PersiaError> {
        let clients = self.clients.read();
        let futs = clients.iter().map(|client| {
            let src_dir = src_dir.clone();
            async move { client.1.load(&src_dir).await }
        });

        let results = futures::future::try_join_all(futs).await?;

        for res in results {
            res?;
        }

        Ok(())
    }

    pub async fn wait_for_serving(&self) -> Result<(), PersiaError> {
        let client = self.get_first_client().clone();

        loop {
            if let Ok(ready) = client.ready_for_serving(&()).await {
                if ready {
                    return Ok(());
                }
            } else {
                tracing::warn!("failed to get embedding model status, retry later");
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
                let status: Vec<EmbeddingModelManagerStatus> =
                    client.model_manager_status(&()).await.unwrap();

                match self.process_status(status) {
                    Ok(_) => {}
                    Err(err_msg) => {
                        return Err(PersiaError::ServerStatusError(err_msg));
                    }
                }
            } else {
                tracing::warn!("failed to get embedding model status, retry later");
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
                            "shutdown embedding worker failed",
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

    pub async fn wait_for_emb_dumping(&self) -> Result<(), PersiaError> {
        let client = self.get_first_client().clone();

        loop {
            std::thread::sleep(Duration::from_secs(5));
            let status: Result<Vec<EmbeddingModelManagerStatus>, _> =
                client.model_manager_status(&()).await;
            if let Ok(status) = status {
                if status.iter().any(|s| match s {
                    EmbeddingModelManagerStatus::Loading(_) => true,
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
                tracing::warn!("failed to get embedding model status, retry later");
            }
        }
    }

    fn process_status(&self, status: Vec<EmbeddingModelManagerStatus>) -> Result<usize, String> {
        let mut num_compeleted: usize = 0;
        let mut errors = Vec::new();
        status
            .into_iter()
            .enumerate()
            .for_each(|(replica_index, s)| match s {
                EmbeddingModelManagerStatus::Failed(e) => {
                    let err_msg = format!(
                        "emb dump FAILED for server {}, due to {:?}.",
                        replica_index, e
                    );
                    errors.push(err_msg);
                }
                EmbeddingModelManagerStatus::Loading(p) => {
                    tracing::info!(
                        "loading emb for server {}, pregress: {:?}%",
                        replica_index,
                        p * 100.0
                    );
                }
                EmbeddingModelManagerStatus::Idle => {
                    num_compeleted = num_compeleted + 1;
                }
                EmbeddingModelManagerStatus::Dumping(p) => {
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
