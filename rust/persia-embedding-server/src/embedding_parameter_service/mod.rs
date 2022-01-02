use std::path::PathBuf;
use std::sync::Arc;

use persia_libs::{
    async_lock, bytes, hyper, lz4, once_cell, rand, rand::Rng, thiserror, tokio,
    tracing,
};
use snafu::ResultExt;

use persia_common::optim::{Optimizable, Optimizer, OptimizerConfig};
use persia_embedding_config::{
    EmbeddinHyperparameters, EmbeddingConfig, EmbeddingParameterServerConfig, InstanceInfo,
    PerisaJobType, PersiaCommonConfig, PersiaGlobalConfigError, PersiaReplicaInfo,
};
use persia_embedding_map::{
    emb_entry::DynamicEmbeddingEntry, EmbeddingShardedMap, EmbeddingShardedMapError,
};
use persia_incremental_update_manager::PerisaIncrementalUpdateManager;

use persia_metrics::{Gauge, IntCounter, PersiaMetricsManager, PersiaMetricsManagerError};
use persia_model_manager::{
    EmbeddingModelManager, EmbeddingModelManagerError, EmbeddingModelManagerStatus,
};
use persia_nats_client::{NatsClient, NatsError};
use persia_speedy::{Readable, Writable};

static METRICS_HOLDER: once_cell::sync::OnceCell<MetricsHolder> = once_cell::sync::OnceCell::new();

struct MetricsHolder {
    pub index_miss_count: IntCounter,
    pub index_miss_ratio: Gauge,
    pub set_embedding_time_cost_sec: Gauge,
    pub lookup_hashmap_time_cost_sec: Gauge,
    pub gradient_id_miss_count: IntCounter,
}

impl MetricsHolder {
    pub fn get() -> Result<&'static Self, PersiaMetricsManagerError> {
        METRICS_HOLDER.get_or_try_init(|| {
            let m = PersiaMetricsManager::get()?;
            let holder = Self {
                index_miss_count: m.create_counter(
                    "index_miss_count",
                    "miss count of index when lookup on embedding server",
                )?,
                index_miss_ratio: m.create_gauge(
                    "index_miss_ratio",
                    "miss ratio of index when lookup on embedding server",
                )?,
                set_embedding_time_cost_sec: m.create_gauge(
                    "set_embedding_time_cost_sec",
                    "set embedding time cost on embedding server",
                )?,
                lookup_hashmap_time_cost_sec: m.create_gauge(
                    "lookup_hashmap_time_cost_sec",
                    "time cost of embedding lookup on embedding server, mainly spent on looking up from hash table.",
                )?,
                gradient_id_miss_count: m.create_counter(
                    "gradient_id_miss_count",
                    "num of embedding not found when update corresponding gradient in a batch",
                )?,
            };
            Ok(holder)
        })
    }
}

#[derive(thiserror::Error, Debug, Readable, Writable)]
pub enum EmbeddingParameterServerError {
    #[error("rpc error")]
    RpcError(String),
    #[error("shutdown error: shutdown channel send signal failed")]
    ShutdownError,
    #[error("service not yet ready error")]
    NotReadyError,
    #[error("service not configured error")]
    NotConfiguredError,
    #[error("model manager error: {0}")]
    EmbeddingModelManagerError(#[from] EmbeddingModelManagerError),
    #[error("nats error: {0}")]
    NatsError(#[from] NatsError),
    #[error("optimizer not found error")]
    OptimizerNotFoundError,
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("embedding dim not match")]
    EmbeddingDimNotMatch,
    #[error("embedding map error: {0}")]
    EmbeddingShardedMapError(#[from] EmbeddingShardedMapError),
}

pub struct EmbeddingParameterServiceInner {
    pub optimizer: async_lock::RwLock<Option<Arc<Box<dyn Optimizable + Send + Sync>>>>,
    pub hyperparameter_config: async_lock::RwLock<Option<Arc<EmbeddinHyperparameters>>>,
    pub hyperparameter_configured: async_lock::Mutex<bool>,
    pub server_config: Arc<EmbeddingParameterServerConfig>,
    pub common_config: Arc<PersiaCommonConfig>,
    pub embedding_config: Arc<EmbeddingConfig>,
    pub inc_update_manager: Arc<PerisaIncrementalUpdateManager>,
    pub embedding_model_manager: Arc<EmbeddingModelManager>,
    pub replica_index: usize,
}

impl EmbeddingParameterServiceInner {
    pub fn new(
        server_config: Arc<EmbeddingParameterServerConfig>,
        common_config: Arc<PersiaCommonConfig>,
        embedding_config: Arc<EmbeddingConfig>,
        inc_update_manager: Arc<PerisaIncrementalUpdateManager>,
        embedding_model_manager: Arc<EmbeddingModelManager>,
        replica_index: usize,
    ) -> Self {
        Self {
            optimizer: async_lock::RwLock::new(None),
            hyperparameter_config: async_lock::RwLock::new(None),
            hyperparameter_configured: async_lock::Mutex::new(false),
            server_config,
            common_config,
            embedding_config,
            inc_update_manager,
            embedding_model_manager,
            replica_index,
        }
    }

    pub fn replica_index(&self) -> usize {
        self.replica_index
    }

    pub fn get_job_type(&self) -> Result<PerisaJobType, EmbeddingParameterServerError> {
        let job_type = self.common_config.job_type.clone();
        Ok(job_type)
    }

    pub async fn get_configuration(
        &self,
    ) -> Result<Arc<EmbeddinHyperparameters>, EmbeddingParameterServerError> {
        let conf = self.hyperparameter_config.read().await;
        let conf = conf.as_ref();
        if let Some(conf) = conf {
            Ok(conf.clone())
        } else {
            Err(EmbeddingParameterServerError::NotConfiguredError)
        }
    }

    pub async fn batched_lookup(
        &self,
        req: Vec<(u64, usize)>,
        is_training: bool,
    ) -> Result<Vec<f32>, EmbeddingParameterServerError> {
        let num_elements: usize = req.iter().map(|x| x.1).sum();
        let mut embeddings = Vec::with_capacity(num_elements);

        let mut index_miss_count: u64 = 0;

        let conf = match is_training {
            true => Some(self.get_configuration().await?),
            false => None,
        };

        let optimizer = self.optimizer.read().await;
        let embedding_map = EmbeddingShardedMap::get()?;

        tokio::task::block_in_place(|| match is_training {
            true => {
                if optimizer.is_none() {
                    return Err(EmbeddingParameterServerError::OptimizerNotFoundError);
                }
                let optimizer = optimizer.as_ref().unwrap();

                req.iter().for_each(|(sign, slot_index)| {
                    let conf = conf.as_ref().unwrap();
                    let slot_config = self.embedding_config.get_slot_by_index(*slot_index);

                    let mut shard = embedding_map.shard(sign).write();

                    match shard.get_refresh(&sign) {
                        None => {
                            if rand::thread_rng().gen_range(0f32..1f32) < conf.admit_probability {
                                let emb = shard.insert_init(*sign, *slot_index, optimizer.clone());
                                embeddings.extend_from_slice(emb.emb());

                                index_miss_count += 1;
                            } else {
                                embeddings
                                    .extend_from_slice(vec![0f32; slot_config.dim].as_slice());
                            }
                        }
                        Some(entry) => {
                            embeddings.extend_from_slice(entry.emb());
                        }
                    }
                });
                Ok(())
            }
            false => {
                req.iter().for_each(|(sign, dim)| {
                    let shard = embedding_map.shard(sign).read();
                    match shard.get(sign) {
                        Some(entry) => {
                            embeddings.extend_from_slice(entry.emb());
                        }
                        None => {
                            embeddings.extend_from_slice(vec![0f32; *dim].as_slice());
                            index_miss_count += 1;
                        }
                    }
                });
                Ok(())
            }
        })?;

        if let Ok(m) = MetricsHolder::get() {
            m.index_miss_count.inc_by(index_miss_count);
            let index_miss_ratio = index_miss_count as f32 / req.len() as f32;
            m.index_miss_ratio.set(index_miss_ratio.into());
        }

        return Ok(embeddings);
    }

    pub async fn ready_for_serving(&self) -> bool {
        let model_status = self.embedding_model_manager.get_status();
        let model_ready = match model_status {
            EmbeddingModelManagerStatus::Dumping(_) => true,
            EmbeddingModelManagerStatus::Idle => true,
            EmbeddingModelManagerStatus::Loading(_) => false,
            EmbeddingModelManagerStatus::Failed(_) => false,
        };
        if !model_ready {
            return false;
        }
        let job_type = self.common_config.job_type.clone();
        match job_type {
            PerisaJobType::Infer => true,
            _ => *self.hyperparameter_configured.lock().await,
        }
    }

    pub async fn model_manager_status(&self) -> EmbeddingModelManagerStatus {
        let status = self.embedding_model_manager.get_status();
        status
    }

    pub async fn set_embedding(
        &self,
        embeddings: Vec<DynamicEmbeddingEntry>,
    ) -> Result<(), EmbeddingParameterServerError> {
        let start_time = std::time::Instant::now();
        let embedding_map = EmbeddingShardedMap::get()?;

        tokio::task::block_in_place(|| {
            embeddings.into_iter().for_each(|entry| {
                let sign = entry.sign;
                let mut shard = embedding_map.shard(&sign).write();
                shard.insert_dyn(sign, entry);
            });
        });

        if let Ok(m) = MetricsHolder::get() {
            m.set_embedding_time_cost_sec
                .set(start_time.elapsed().as_secs_f64());
        }
        Ok(())
    }

    pub async fn lookup_mixed(
        &self,
        req: (Vec<(u64, usize)>, bool),
    ) -> Result<Vec<f32>, EmbeddingParameterServerError> {
        let (indices, is_training) = req;
        let start_time = std::time::Instant::now();
        let embedding = self.batched_lookup(indices, is_training).await;
        if let Ok(m) = MetricsHolder::get() {
            m.lookup_hashmap_time_cost_sec
                .set(start_time.elapsed().as_secs_f64());
        }

        embedding
    }

    pub async fn update_gradient_mixed(
        &self,
        req: (Vec<u64>, Vec<f32>),
    ) -> Result<(), EmbeddingParameterServerError> {
        let conf = self.get_configuration().await?;
        let (signs, remaining_gradients) = req;
        let mut remaining_gradients = remaining_gradients.as_slice();
        let mut indices_to_commit = Vec::with_capacity(signs.len());
        let mut gradient_id_miss_count = 0;

        let optimizer = self.optimizer.read().await;
        if optimizer.is_none() {
            return Err(EmbeddingParameterServerError::OptimizerNotFoundError);
        }

        let optimizer = optimizer.as_ref().unwrap();
        let embedding_map = EmbeddingShardedMap::get()?;

        tokio::task::block_in_place(|| {
            for sign in signs.iter() {
                let mut shard = embedding_map.shard(sign).write();
                if let Some(mut entry) = shard.get_mut(sign) {
                    let entry_dim = entry.embedding_dim;
                    let (grad, r) = remaining_gradients.split_at(entry_dim);
                    remaining_gradients = r;

                    {
                        optimizer.update(entry.inner, grad, entry_dim);
                        if conf.enable_weight_bound {
                            unsafe {
                                persia_simd::weight_bound(entry.emb(), conf.weight_bound);
                            }
                        }
                    }

                    indices_to_commit.push(*sign);
                } else {
                    gradient_id_miss_count += 1;
                }
            }
        });

        tracing::debug!(
            "Gradient update failed {} times due to embedding not found",
            gradient_id_miss_count
        );
        if let Ok(m) = MetricsHolder::get() {
            m.gradient_id_miss_count.inc_by(gradient_id_miss_count);
        }

        if self.server_config.enable_incremental_update {
            let result = self
                .inc_update_manager
                .try_commit_incremental(indices_to_commit);
            if result.is_err() {
                tracing::warn!(
                    "inc update failed, please try a bigger inc_update_sending_buffer_size"
                );
            }
        }

        Ok(())
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), EmbeddingParameterServerError> {
        {
            let mut optimizer_ = self.optimizer.write().await;
            *optimizer_ = Some(Arc::new(Optimizer::new(optimizer).to_optimizable()));
        }

        let optimizer = self.optimizer.read().await.as_ref().unwrap().clone();

        if let Some(config) = self.hyperparameter_config.read().await.as_ref() {
            let _ = EmbeddingShardedMap::set(optimizer, config.as_ref().clone());
        }
        Ok(())
    }

    pub async fn configure(
        &self,
        config: EmbeddinHyperparameters,
    ) -> Result<(), EmbeddingParameterServerError> {
        {
            let mut conf_guard = self.hyperparameter_config.write().await;
            *conf_guard = Some(Arc::new(config.clone()));
        }
        *self.hyperparameter_configured.lock().await = true;
        tracing::info!("embedding server configured");

        if let Some(optimizer) = self.optimizer.read().await.as_ref() {
            let _ = EmbeddingShardedMap::set(optimizer.clone(), config);
        }

        Ok(())
    }

    pub async fn dump(&self, dir: String) -> Result<(), EmbeddingParameterServerError> {
        let embedding_map = EmbeddingShardedMap::get()?;
        let dst_dir = PathBuf::from(dir);
        self.embedding_model_manager
            .dump_embedding(dst_dir, embedding_map)?;
        Ok(())
    }

    pub async fn load(&self, dir: String) -> Result<(), EmbeddingParameterServerError> {
        let embedding_map = EmbeddingShardedMap::get()?;
        let dst_dir = PathBuf::from(dir);
        let shard_dir = self.embedding_model_manager.get_shard_dir(&dst_dir);
        self.embedding_model_manager
            .load_embedding_from_dir(shard_dir, embedding_map)?;
        Ok(())
    }

    pub async fn get_address(&self) -> Result<String, EmbeddingParameterServerError> {
        let instance_info = InstanceInfo::get()?;
        let address = format!("{}:{}", instance_info.ip_address, instance_info.port);
        Ok(address)
    }

    pub async fn get_replica_info(
        &self,
    ) -> Result<PersiaReplicaInfo, EmbeddingParameterServerError> {
        let replica_info = PersiaReplicaInfo::get()?;
        let replica_info = replica_info.as_ref().clone();
        Ok(replica_info)
    }

    pub async fn get_embedding_size(&self) -> Result<usize, EmbeddingParameterServerError> {
        let embedding_map = EmbeddingShardedMap::get()?;
        Ok(embedding_map.num_total_signs())
    }

    pub async fn clear_embeddings(&self) -> Result<(), EmbeddingParameterServerError> {
        let embedding_map = EmbeddingShardedMap::get()?;
        Ok(embedding_map.clear())
    }
}

#[derive(Clone)]
pub struct EmbeddingParameterService {
    pub inner: Arc<EmbeddingParameterServiceInner>,
    pub shutdown_channel: Arc<async_lock::RwLock<Option<tokio::sync::oneshot::Sender<()>>>>,
}

#[persia_rpc_macro::service]
impl EmbeddingParameterService {
    pub async fn ready_for_serving(&self, _req: ()) -> bool {
        self.inner.ready_for_serving().await
    }

    pub async fn model_manager_status(&self, _req: ()) -> EmbeddingModelManagerStatus {
        self.inner.model_manager_status().await
    }

    pub async fn set_embedding(
        &self,
        req: Vec<DynamicEmbeddingEntry>,
    ) -> Result<(), EmbeddingParameterServerError> {
        self.inner.set_embedding(req).await
    }

    pub async fn lookup_mixed(
        &self,
        req: (Vec<(u64, usize)>, bool),
    ) -> Result<Vec<f32>, EmbeddingParameterServerError> {
        self.inner.lookup_mixed(req).await
    }

    pub async fn replica_index(&self, _req: ()) -> usize {
        self.inner.replica_index()
    }

    pub async fn update_gradient_mixed(
        &self,
        req: (Vec<u64>, Vec<f32>),
    ) -> Result<(), EmbeddingParameterServerError> {
        self.inner.update_gradient_mixed(req).await
    }

    pub async fn configure(
        &self,
        config: EmbeddinHyperparameters,
    ) -> Result<(), EmbeddingParameterServerError> {
        self.inner.configure(config).await
    }

    pub async fn dump(&self, req: String) -> Result<(), EmbeddingParameterServerError> {
        self.inner.dump(req).await
    }

    pub async fn load(&self, req: String) -> Result<(), EmbeddingParameterServerError> {
        self.inner.load(req).await
    }

    pub async fn get_embedding_size(
        &self,
        _req: (),
    ) -> Result<usize, EmbeddingParameterServerError> {
        self.inner.get_embedding_size().await
    }

    pub async fn clear_embeddings(&self, _req: ()) -> Result<(), EmbeddingParameterServerError> {
        self.inner.clear_embeddings().await
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), EmbeddingParameterServerError> {
        self.inner.register_optimizer(optimizer).await
    }

    pub async fn shutdown(&self, _req: ()) -> Result<(), EmbeddingParameterServerError> {
        let mut shutdown_channel = self.shutdown_channel.write().await;
        let shutdown_channel = shutdown_channel.take();
        match shutdown_channel {
            Some(sender) => match sender.send(()) {
                Ok(_) => {
                    tracing::info!("receive shutdown signal, shutdown the server after processed the remain requests.");
                    Ok(())
                }
                Err(_) => {
                    tracing::warn!("Send the shutdown singal failed corresponding receiver has already been deallocated");
                    Err(EmbeddingParameterServerError::ShutdownError)
                }
            },
            None => {
                tracing::debug!("shutdown channel already been taken, wait server shutdown.");
                Ok(())
            }
        }
    }
}

#[derive(Clone)]
pub struct EmbeddingParameterNatsService {
    pub inner: Arc<EmbeddingParameterServiceInner>,
}

#[persia_nats_marcos::service]
impl EmbeddingParameterNatsService {
    pub async fn ready_for_serving(&self, _req: ()) -> bool {
        self.inner.ready_for_serving().await
    }

    pub async fn model_manager_status(&self, _req: ()) -> EmbeddingModelManagerStatus {
        self.inner.model_manager_status().await
    }

    pub async fn replica_index(&self, _req: ()) -> usize {
        self.inner.replica_index()
    }

    pub async fn configure(
        &self,
        config: EmbeddinHyperparameters,
    ) -> Result<(), EmbeddingParameterServerError> {
        self.inner.configure(config).await
    }

    pub async fn dump(&self, req: String) -> Result<(), EmbeddingParameterServerError> {
        self.inner.dump(req).await
    }

    pub async fn load(&self, req: String) -> Result<(), EmbeddingParameterServerError> {
        self.inner.load(req).await
    }

    pub async fn get_address(&self, _req: ()) -> Result<String, EmbeddingParameterServerError> {
        self.inner.get_address().await
    }

    pub async fn get_replica_info(
        &self,
        _req: (),
    ) -> Result<PersiaReplicaInfo, EmbeddingParameterServerError> {
        self.inner.get_replica_info().await
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), EmbeddingParameterServerError> {
        self.inner.register_optimizer(optimizer).await
    }
}
