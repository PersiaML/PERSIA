use std::path::PathBuf;
use std::sync::Arc;

use persia_libs::{
    bytes, bytes::Bytes, hyper, lz4, once_cell, parking_lot, rand, rand::Rng, thiserror, tokio,
    tracing,
};
use snafu::ResultExt;

use persia_common::{
    optim::{Optimizable, Optimizer, OptimizerConfig},
    HashMapEmbeddingEntry,
};
use persia_embedding_config::{
    EmbeddingConfig, InstanceInfo, PerisaJobType, PersiaCommonConfig, PersiaEmbeddingServerConfig,
    PersiaGlobalConfigError, PersiaReplicaInfo, PersiaSparseModelHyperparameters,
};
use persia_embedding_holder::PersiaEmbeddingHolder;
use persia_full_amount_manager::FullAmountManager;
use persia_incremental_update_manager::PerisaIncrementalUpdateManager;

use persia_metrics::{
    Gauge, Histogram, IntCounter, PersiaMetricsManager, PersiaMetricsManagerError,
};
use persia_model_manager::{
    PersiaPersistenceManager, PersiaPersistenceStatus, PersistenceManagerError,
};
use persia_nats_client::{NatsClient, NatsError};
use persia_speedy::{Readable, Writable};

static METRICS_HOLDER: once_cell::sync::OnceCell<MetricsHolder> = once_cell::sync::OnceCell::new();

struct MetricsHolder {
    pub index_miss_count: IntCounter,
    pub index_miss_ratio: Gauge,
    pub set_embedding_time_cost: Histogram,
    pub decode_indices_time_cost: Histogram,
    pub encode_embedding_time_cost: Histogram,
    pub lookup_inference_batch_time_cost: Histogram,
    pub lookup_mixed_batch_time_cost: Histogram,
    pub gradient_id_miss_count: IntCounter,
}

impl MetricsHolder {
    pub fn get() -> Result<&'static Self, PersiaMetricsManagerError> {
        METRICS_HOLDER.get_or_try_init(|| {
            let m = PersiaMetricsManager::get()?;
            let holder = Self {
                index_miss_count: m
                    .create_counter("index_miss_count", "miss count of index when lookup")?,
                index_miss_ratio: m
                    .create_gauge("index_miss_ratio", "miss ratio of index when lookup")?,
                set_embedding_time_cost: m.create_histogram("set_embedding_time_cost", "ATT")?,
                decode_indices_time_cost: m.create_histogram("decode_indices_time_cost", "ATT")?,
                encode_embedding_time_cost: m
                    .create_histogram("encode_embedding_time_cost", "ATT")?,
                lookup_inference_batch_time_cost: m
                    .create_histogram("lookup_inference_batch_time_cost", "ATT")?,
                lookup_mixed_batch_time_cost: m
                    .create_histogram("lookup_mixed_batch_time_cost", "ATT")?,
                gradient_id_miss_count: m.create_counter("gradient_id_miss_count", "ATT")?,
            };
            Ok(holder)
        })
    }
}

#[derive(thiserror::Error, Debug, Readable, Writable)]
pub enum EmbeddingServerError {
    #[error("rpc error")]
    RpcError(String),
    #[error("shutdown error: shutdown channel send signal failed")]
    ShutdownError,
    #[error("service not yet ready error")]
    NotReadyError,
    #[error("service not configured error")]
    NotConfiguredError,
    #[error("model manager error: {0}")]
    PersistenceManagerError(#[from] PersistenceManagerError),
    #[error("nats error: {0}")]
    NatsError(#[from] NatsError),
    #[error("optimizer not found error")]
    OptimizerNotFoundError,
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("embedding dim not match")]
    EmbeddingDimNotMatch,
}

pub struct EmbeddingServiceInner {
    pub embedding: PersiaEmbeddingHolder,
    pub optimizer: persia_libs::async_lock::RwLock<Option<Arc<Box<dyn Optimizable + Send + Sync>>>>,
    pub hyperparameter_config:
        persia_libs::async_lock::RwLock<Option<Arc<PersiaSparseModelHyperparameters>>>,
    pub hyperparameter_configured: persia_libs::async_lock::Mutex<bool>,
    pub server_config: Arc<PersiaEmbeddingServerConfig>,
    pub common_config: Arc<PersiaCommonConfig>,
    pub embedding_config: Arc<EmbeddingConfig>,
    pub inc_update_manager: Arc<PerisaIncrementalUpdateManager>,
    pub model_persistence_manager: Arc<PersiaPersistenceManager>,
    pub full_amount_manager: Arc<FullAmountManager>,
    pub replica_index: usize,
}

impl EmbeddingServiceInner {
    pub fn new(
        embedding: PersiaEmbeddingHolder,
        server_config: Arc<PersiaEmbeddingServerConfig>,
        common_config: Arc<PersiaCommonConfig>,
        embedding_config: Arc<EmbeddingConfig>,
        inc_update_manager: Arc<PerisaIncrementalUpdateManager>,
        model_persistence_manager: Arc<PersiaPersistenceManager>,
        full_amount_manager: Arc<FullAmountManager>,
        replica_index: usize,
    ) -> Self {
        Self {
            embedding,
            optimizer: persia_libs::async_lock::RwLock::new(None),
            hyperparameter_config: persia_libs::async_lock::RwLock::new(None),
            hyperparameter_configured: persia_libs::async_lock::Mutex::new(false),
            server_config,
            common_config,
            embedding_config,
            inc_update_manager,
            model_persistence_manager,
            full_amount_manager,
            replica_index,
        }
    }

    pub fn replica_index(&self) -> usize {
        self.replica_index
    }

    pub fn get_job_type(&self) -> Result<PerisaJobType, EmbeddingServerError> {
        let job_type = self.common_config.job_type.clone();
        Ok(job_type)
    }

    pub async fn get_configuration(
        &self,
    ) -> Result<Arc<PersiaSparseModelHyperparameters>, EmbeddingServerError> {
        let conf = self.hyperparameter_config.read().await;
        let conf = conf.as_ref();
        if let Some(conf) = conf {
            Ok(conf.clone())
        } else {
            Err(EmbeddingServerError::NotConfiguredError)
        }
    }

    pub async fn batched_lookup(
        &self,
        req: Vec<(u64, usize)>,
        is_training: bool,
    ) -> Result<Vec<f32>, EmbeddingServerError> {
        let num_elements: usize = req.iter().map(|x| x.1).sum();
        let mut embeddings = Vec::with_capacity(num_elements);

        let mut index_miss_count: usize = 0;

        let conf = match is_training {
            true => Some(self.get_configuration().await?),
            false => None,
        };

        let optimizer = self.optimizer.read().await;

        tokio::task::block_in_place(|| match is_training {
            true => {
                if optimizer.is_none() {
                    return Err(EmbeddingServerError::OptimizerNotFoundError);
                }
                let optimizer = optimizer.as_ref().unwrap();

                let mut evcited_ids = Vec::with_capacity(req.len());

                req.iter().for_each(|(sign, dim)| {
                        let conf = conf.as_ref().unwrap();
                        let e = self.embedding.get_value_refresh(sign);
                        match e {
                            None => {
                                if rand::thread_rng().gen_range(0f32..1f32) < conf.admit_probability {
                                    let mut emb_entry = HashMapEmbeddingEntry::new(
                                        &conf.initialization_method,
                                        *dim,
                                        optimizer.require_space(*dim),
                                        *sign,
                                    );

                                    optimizer.state_initialization(emb_entry.as_mut_emb_entry_slice(), *dim);
                                    embeddings.extend_from_slice(&emb_entry.as_emb_entry_slice()[..*dim]);
                                    let evcited = self.embedding
                                        .insert(*sign, Arc::new(parking_lot::RwLock::new(emb_entry)));

                                    if evcited.is_some() {
                                        evcited_ids.push(sign.clone());
                                    }

                                    index_miss_count += 1;
                                } else {
                                    embeddings.extend_from_slice(vec![0f32; *dim].as_slice());
                                }
                            }
                            Some(entry) => {
                                if let Some(entry) = entry.upgrade() {
                                    let entry_dim = { entry.read().dim() };
                                    if entry_dim != *dim {
                                        tracing::error!("dimensional mismatch on sign {}, in hashmap dim {}, requested dim {}", sign, entry_dim, dim);
                                        let entry = HashMapEmbeddingEntry::new(
                                            &conf.initialization_method,
                                            *dim,
                                            0,
                                            *sign,
                                        );
                                        embeddings.extend_from_slice(entry.emb());
                                        let evcited = self
                                            .embedding
                                            .insert(*sign, Arc::new(parking_lot::RwLock::new(entry)));
                                        if evcited.is_some() {
                                            evcited_ids.push(sign.clone());
                                        }
                                    } else {
                                        embeddings.extend_from_slice(entry.read().emb());
                                    }
                                }
                            }
                        }
                    });
                if let Err(_) = self.full_amount_manager.try_commit_evicted_ids(evcited_ids) {
                    tracing::warn!(
                            "commit to full_amount_manager failed, it is ok when dumping emb, otherwise, 
                            please try a bigger full_amount_manager_buffer_size or num_hashmap_internal_shards"
                        );
                }
                Ok(())
            }
            false => {
                req.iter().for_each(|(sign, dim)| {
                        let e = self.embedding.get_value(sign);
                        match e {
                            None => {
                                embeddings.extend_from_slice(vec![0f32; *dim].as_slice());
                                index_miss_count += 1;
                            }
                            Some(entry) => {
                                if let Some(entry) = entry.upgrade() {
                                    let entry_dim = { entry.read().dim() };
                                    if entry_dim != *dim {
                                        tracing::error!("dimensional mismatch on sign {}, in hashmap dim {}, requested dim {}",
                                            sign, entry_dim, dim);
                                        embeddings.extend_from_slice(vec![0f32; *dim].as_slice());
                                    } else {
                                        embeddings.extend_from_slice(entry.read().emb());
                                    }
                                }
                            }
                        }
                    });
                Ok(())
            }
        })?;

        if let Ok(m) = MetricsHolder::get() {
            m.index_miss_count.inc();
            let index_miss_ratio = index_miss_count as f32 / req.len() as f32;
            m.index_miss_ratio.set(index_miss_ratio.into());
        }

        return Ok(embeddings);
    }

    pub async fn ready_for_serving(&self) -> bool {
        let model_status = self.model_persistence_manager.get_status();
        let model_ready = match model_status {
            PersiaPersistenceStatus::Dumping(_) => true,
            PersiaPersistenceStatus::Idle => true,
            PersiaPersistenceStatus::Loading(_) => false,
            PersiaPersistenceStatus::Failed(_) => false,
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

    pub async fn model_manager_status(&self) -> PersiaPersistenceStatus {
        let status = self.model_persistence_manager.get_status();
        status
    }

    pub async fn set_embedding(
        &self,
        embeddings: Vec<(u64, HashMapEmbeddingEntry)>,
    ) -> Result<(), EmbeddingServerError> {
        let start_time = std::time::Instant::now();
        let mut evcited_ids = Vec::with_capacity(embeddings.len());
        tokio::task::block_in_place(|| {
            embeddings.into_iter().for_each(|(id, entry)| {
                let evcited = self
                    .embedding
                    .insert(id, Arc::new(parking_lot::RwLock::new(entry)));
                if evcited.is_some() {
                    evcited_ids.push(id);
                }
            });
        });
        if let Err(_) = self.full_amount_manager.try_commit_evicted_ids(evcited_ids) {
            tracing::warn!(
                    "commit to full_amount_manager failed, it is ok when dumping emb, otherwise, 
                    please try a bigger full_amount_manager_buffer_size or num_hashmap_internal_shards"
                );
        }
        if let Ok(m) = MetricsHolder::get() {
            m.set_embedding_time_cost
                .observe(start_time.elapsed().as_secs_f64());
        }
        Ok(())
    }

    pub async fn lookup_inference(&self, req: Bytes) -> Result<Bytes, EmbeddingServerError> {
        let start_time = std::time::Instant::now();
        let indices =
            tokio::task::block_in_place(|| Vec::<(u64, usize)>::read_from_buffer(req.as_ref()));
        if indices.is_err() {
            return Err(EmbeddingServerError::RpcError(
                "fail to des request".to_string(),
            ));
        }
        let indices = indices.unwrap();
        if let Ok(m) = MetricsHolder::get() {
            m.decode_indices_time_cost
                .observe(start_time.elapsed().as_secs_f64());
        }

        let embedding = self.batched_lookup(indices, false).await;
        if let Ok(emb) = embedding {
            let encode_start_time = std::time::Instant::now();
            let buffer = tokio::task::block_in_place(|| emb.write_to_vec().unwrap());
            if let Ok(m) = MetricsHolder::get() {
                m.encode_embedding_time_cost
                    .observe(encode_start_time.elapsed().as_secs_f64());
                m.lookup_inference_batch_time_cost
                    .observe(start_time.elapsed().as_secs_f64());
            }
            Ok(Bytes::from(buffer))
        } else {
            Err(EmbeddingServerError::RpcError(
                "fail to lookup embedding".to_string(),
            ))
        }
    }

    pub async fn lookup_mixed(
        &self,
        req: (Vec<(u64, usize)>, bool),
    ) -> Result<Vec<f32>, EmbeddingServerError> {
        let (indices, is_training) = req;
        let start_time = std::time::Instant::now();
        let embedding = self.batched_lookup(indices, is_training).await;
        if let Ok(m) = MetricsHolder::get() {
            m.lookup_mixed_batch_time_cost
                .observe(start_time.elapsed().as_secs_f64());
        }

        embedding
    }

    pub async fn update_gradient_mixed(
        &self,
        req: (Vec<u64>, Vec<f32>),
    ) -> Result<(), EmbeddingServerError> {
        let conf = self.get_configuration().await?;
        let (signs, remaining_gradients) = req;
        let mut remaining_gradients = remaining_gradients.as_slice();
        let mut indices_to_commit = Vec::with_capacity(signs.len());
        let mut gradient_id_miss_count = 0;

        let optimizer = self.optimizer.read().await;
        if optimizer.is_none() {
            return Err(EmbeddingServerError::OptimizerNotFoundError);
        }

        let optimizer = optimizer.as_ref().unwrap();
        let batch_level_state = optimizer.get_batch_level_state(signs.as_slice());

        tokio::task::block_in_place(|| {
            for (idx, sign) in signs.iter().enumerate() {
                if let Some(entry) = self.embedding.get_value(&sign) {
                    if let Some(entry) = entry.upgrade() {
                        let entry_dim = { entry.read().dim() };
                        let (grad, r) = remaining_gradients.split_at(entry_dim);
                        remaining_gradients = r;
                        let emb_opt_state = optimizer.get_emb_state(&batch_level_state, idx);

                        {
                            let mut entry = entry.write();
                            let emb_entry_slice = entry.as_mut_emb_entry_slice();
                            optimizer.update(emb_entry_slice, grad, entry_dim, &emb_opt_state);

                            if conf.enable_weight_bound {
                                unsafe {
                                    persia_simd::weight_bound(
                                        &mut emb_entry_slice[..entry_dim],
                                        conf.weight_bound,
                                    );
                                }
                            }
                        }

                        indices_to_commit.push((*sign, entry.clone()));
                    }
                } else {
                    gradient_id_miss_count += 1;
                }
            }
        });

        tracing::debug!(
            "{} update gradient corresponding embedding not found, skipped",
            gradient_id_miss_count
        );
        if let Ok(m) = MetricsHolder::get() {
            m.gradient_id_miss_count.inc_by(gradient_id_miss_count);
        }

        let weak_ptrs = indices_to_commit
            .iter()
            .map(|(k, v)| (k.clone(), Arc::downgrade(v)))
            .collect();
        let commit_result = self.full_amount_manager.try_commit_weak_ptrs(weak_ptrs);
        if commit_result.is_err() {
            tracing::warn!(
                "commit to full_amount_manager failed, it is ok when dumping emb, otherwise, 
                please try a bigger full_amount_manager_buffer_size or num_hashmap_internal_shards"
            );
        }

        if self.server_config.enable_incremental_update {
            let result = self
                .inc_update_manager
                .try_commit_incremental(indices_to_commit);
            if result.is_err() {
                tracing::error!(
                    "inc update failed, please try a bigger inc_update_sending_buffer_size"
                );
            }
        }

        Ok(())
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), EmbeddingServerError> {
        {
            let mut optimizer_ = self.optimizer.write().await;
            *optimizer_ = Some(Arc::new(Optimizer::new(optimizer).to_optimizable()));
        }
        Ok(())
    }

    pub async fn configure(
        &self,
        config: PersiaSparseModelHyperparameters,
    ) -> Result<(), EmbeddingServerError> {
        {
            let mut conf_guard = self.hyperparameter_config.write().await;
            *conf_guard = Some(Arc::new(config));
        }
        *self.hyperparameter_configured.lock().await = true;
        tracing::info!("embedding server configured");
        Ok(())
    }

    pub async fn dump(&self, dir: String) -> Result<(), EmbeddingServerError> {
        let dst_dir = PathBuf::from(dir);
        self.model_persistence_manager
            .dump_full_amount_embedding(dst_dir)?;
        Ok(())
    }

    pub async fn load(&self, dir: String) -> Result<(), EmbeddingServerError> {
        let dst_dir = PathBuf::from(dir);
        self.model_persistence_manager
            .load_embedding_from_dir(dst_dir)?;
        Ok(())
    }

    pub async fn get_address(&self) -> Result<String, EmbeddingServerError> {
        let instance_info = InstanceInfo::get()?;
        let address = format!("{}:{}", instance_info.ip_address, instance_info.port);
        Ok(address)
    }

    pub async fn get_replica_info(&self) -> Result<PersiaReplicaInfo, EmbeddingServerError> {
        let replica_info = PersiaReplicaInfo::get()?;
        let replica_info = replica_info.as_ref().clone();
        Ok(replica_info)
    }

    pub async fn get_embedding_size(&self) -> Result<usize, EmbeddingServerError> {
        Ok(self.embedding.num_total_signs())
    }

    pub async fn clear_embeddings(&self) -> Result<(), EmbeddingServerError> {
        Ok(self.embedding.clear())
    }
}

#[derive(Clone)]
pub struct EmbeddingService {
    pub inner: Arc<EmbeddingServiceInner>,
    pub shutdown_channel:
        Arc<persia_libs::async_lock::RwLock<Option<tokio::sync::oneshot::Sender<()>>>>,
}

#[persia_rpc_macro::service]
impl EmbeddingService {
    pub async fn ready_for_serving(&self, _req: ()) -> bool {
        self.inner.ready_for_serving().await
    }

    pub async fn model_manager_status(&self, _req: ()) -> PersiaPersistenceStatus {
        self.inner.model_manager_status().await
    }

    pub async fn set_embedding(
        &self,
        req: Vec<(u64, HashMapEmbeddingEntry)>,
    ) -> Result<(), EmbeddingServerError> {
        self.inner.set_embedding(req).await
    }

    pub async fn lookup_inference(&self, req: Bytes) -> Result<Bytes, EmbeddingServerError> {
        self.inner.lookup_inference(req).await
    }

    pub async fn lookup_mixed(
        &self,
        req: (Vec<(u64, usize)>, bool),
    ) -> Result<Vec<f32>, EmbeddingServerError> {
        self.inner.lookup_mixed(req).await
    }

    pub async fn replica_index(&self, _req: ()) -> usize {
        self.inner.replica_index()
    }

    pub async fn update_gradient_mixed(
        &self,
        req: (Vec<u64>, Vec<f32>),
    ) -> Result<(), EmbeddingServerError> {
        self.inner.update_gradient_mixed(req).await
    }

    pub async fn configure(
        &self,
        config: PersiaSparseModelHyperparameters,
    ) -> Result<(), EmbeddingServerError> {
        self.inner.configure(config).await
    }

    pub async fn dump(&self, req: String) -> Result<(), EmbeddingServerError> {
        self.inner.dump(req).await
    }

    pub async fn load(&self, req: String) -> Result<(), EmbeddingServerError> {
        self.inner.load(req).await
    }

    pub async fn get_embedding_size(&self, _req: ()) -> Result<usize, EmbeddingServerError> {
        self.inner.get_embedding_size().await
    }

    pub async fn clear_embeddings(&self, _req: ()) -> Result<(), EmbeddingServerError> {
        self.inner.clear_embeddings().await
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), EmbeddingServerError> {
        self.inner.register_optimizer(optimizer).await
    }

    pub async fn shutdown(&self, _req: ()) -> Result<(), EmbeddingServerError> {
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
                    Err(EmbeddingServerError::ShutdownError)
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
pub struct EmbeddingServerNatsService {
    pub inner: Arc<EmbeddingServiceInner>,
}

#[persia_nats_marcos::service]
impl EmbeddingServerNatsService {
    pub async fn ready_for_serving(&self, _req: ()) -> bool {
        self.inner.ready_for_serving().await
    }

    pub async fn model_manager_status(&self, _req: ()) -> PersiaPersistenceStatus {
        self.inner.model_manager_status().await
    }

    pub async fn replica_index(&self, _req: ()) -> usize {
        self.inner.replica_index()
    }

    pub async fn configure(
        &self,
        config: PersiaSparseModelHyperparameters,
    ) -> Result<(), EmbeddingServerError> {
        self.inner.configure(config).await
    }

    pub async fn dump(&self, req: String) -> Result<(), EmbeddingServerError> {
        self.inner.dump(req).await
    }

    pub async fn load(&self, req: String) -> Result<(), EmbeddingServerError> {
        self.inner.load(req).await
    }

    pub async fn get_address(&self, _req: ()) -> Result<String, EmbeddingServerError> {
        self.inner.get_address().await
    }

    pub async fn get_replica_info(
        &self,
        _req: (),
    ) -> Result<PersiaReplicaInfo, EmbeddingServerError> {
        self.inner.get_replica_info().await
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), EmbeddingServerError> {
        self.inner.register_optimizer(optimizer).await
    }
}
