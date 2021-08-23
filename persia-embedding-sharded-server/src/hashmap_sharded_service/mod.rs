use std::collections::LinkedList;
use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use hashbrown::HashMap;
use rand::Rng;
use snafu::ResultExt;
use thiserror::Error;

use persia_embedding_config::{
    InstanceInfo, PerisaIntent, PersiaCommonConfig, PersiaGlobalConfigError, PersiaReplicaInfo,
    PersiaShardedServerConfig, PersiaSparseModelHyperparameters,
};
use persia_embedding_datatypes::optim::{Optimizable, Optimizer};
use persia_embedding_datatypes::HashMapEmbeddingEntry;
use persia_embedding_holder::PersiaEmbeddingHolder;
use persia_full_amount_manager::FullAmountManager;
use persia_futures::tokio;
use persia_incremental_update_manager::PerisaIncrementalUpdateManager;
use persia_metrics::{
    Gauge, Histogram, IntCounter, PersiaMetricsManager, PersiaMetricsManagerError,
};
use persia_model_manager::{
    PersiaPersistenceManager, PersiaPersistenceStatus, PersistenceManagerError,
};
use persia_nats_client::{NatsClient, NatsError};
use persia_speedy::{Readable, Writable};

static SET_EMBEDDING_POOL: once_cell::sync::OnceCell<EmbeddingPool> =
    once_cell::sync::OnceCell::new();

pub struct EmbeddingPool {
    pub inner: Arc<
        parking_lot::RwLock<
            HashMap<
                usize,
                parking_lot::Mutex<LinkedList<Arc<parking_lot::RwLock<HashMapEmbeddingEntry>>>>,
            >,
        >,
    >,
    pub capacity: usize,
}

impl EmbeddingPool {
    pub fn push(&self, entry: Arc<parking_lot::RwLock<HashMapEmbeddingEntry>>) -> usize {
        let dim = { entry.read().dim_infer() };
        let entry_list_exist = {
            let map = self.inner.read();
            map.get(&dim).is_some()
        };
        if !entry_list_exist {
            let mut map = self.inner.write();
            let list = map.get(&dim);
            if list.is_none() {
                let empty_list = parking_lot::Mutex::new(LinkedList::new());
                map.insert(dim, empty_list);
            }
        }
        let map = self.inner.read();
        let mut list = map.get(&dim).unwrap().lock();
        if Arc::strong_count(&entry) > 1 {
            tracing::error!("entry put into embedding pool is still in use");
            if let Ok(m) = MetricsHolder::get() {
                m.recycle_embedding_still_in_use.inc();
            }
        } else {
            if list.len() < self.capacity {
                list.push_back(entry);
            }
        }
        let cur_len = list.len();

        if let Ok(m) = MetricsHolder::get() {
            m.embedding_pool_push_qps.inc();
            m.embedding_pool_size.set(cur_len as f64);
        }

        cur_len
    }

    pub fn pull(&self, dim: usize) -> Option<Arc<parking_lot::RwLock<HashMapEmbeddingEntry>>> {
        let map = self.inner.read();
        let list = map.get(&dim);
        if list.is_none() {
            return None;
        }
        if let Ok(m) = MetricsHolder::get() {
            m.embedding_pool_pull_qps.inc();
        }
        let mut list = list.unwrap().lock();
        list.pop_back()
    }
}

impl Default for EmbeddingPool {
    fn default() -> Self {
        Self {
            inner: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            capacity: get_embedding_pool_cap(),
        }
    }
}

pub fn get_embedding_pool_cap() -> usize {
    let pool_cap = std::env::var("PERSIA_EMBEDDING_POOL_CAP").unwrap_or("5000".to_string());
    let cap: usize = pool_cap.parse().unwrap_or(5000);
    cap
}

static METRICS_HOLDER: once_cell::sync::OnceCell<MetricsHolder> = once_cell::sync::OnceCell::new();

struct MetricsHolder {
    pub index_miss_count: IntCounter,
    pub index_miss_ratio: Gauge,
    pub set_embedding_time_cost: Histogram,
    pub set_embedding_miss_ratio: Gauge,
    pub set_embedding_recycled_ratio: Gauge,
    pub decode_indices_time_cost: Histogram,
    pub encode_embedding_time_cost: Histogram,
    pub lookup_inference_batch_time_cost: Histogram,
    pub lookup_mixed_batch_time_cost: Histogram,
    pub gradient_id_miss_count: IntCounter,
    pub recycle_embedding_still_in_use: IntCounter,
    pub embedding_pool_push_qps: IntCounter,
    pub embedding_pool_pull_qps: IntCounter,
    pub embedding_pool_size: Gauge,
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
                set_embedding_miss_ratio: m.create_gauge("set_embedding_miss_ratio", "ATT")?,
                set_embedding_recycled_ratio: m
                    .create_gauge("set_embedding_recycled_ratio", "ATT")?,
                decode_indices_time_cost: m.create_histogram("decode_indices_time_cost", "ATT")?,
                encode_embedding_time_cost: m
                    .create_histogram("encode_embedding_time_cost", "ATT")?,
                lookup_inference_batch_time_cost: m
                    .create_histogram("lookup_inference_batch_time_cost", "ATT")?,
                lookup_mixed_batch_time_cost: m
                    .create_histogram("lookup_mixed_batch_time_cost", "ATT")?,
                gradient_id_miss_count: m.create_counter("gradient_id_miss_count", "ATT")?,
                recycle_embedding_still_in_use: m
                    .create_counter("recycle_embedding_still_in_use", "ATT")?,
                embedding_pool_push_qps: m.create_counter("embedding_pool_push_qps", "ATT")?,
                embedding_pool_pull_qps: m.create_counter("embedding_pool_pull_qps", "ATT")?,
                embedding_pool_size: m.create_gauge("embedding_pool_size", "ATT")?,
            };
            Ok(holder)
        })
    }
}

#[derive(Error, Debug, Readable, Writable)]
pub enum ShardEmbeddingError {
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
}

pub struct HashMapShardedServiceInner {
    pub embedding: PersiaEmbeddingHolder,
    pub optimizer:
        persia_futures::async_lock::RwLock<Option<Arc<Box<dyn Optimizable + Send + Sync>>>>,
    pub hyperparameter_config:
        persia_futures::async_lock::RwLock<Option<Arc<PersiaSparseModelHyperparameters>>>,
    pub hyperparameter_configured: persia_futures::async_lock::Mutex<bool>,
    pub server_config: Arc<PersiaShardedServerConfig>,
    pub common_config: Arc<PersiaCommonConfig>,
    pub inc_update_manager: Arc<PerisaIncrementalUpdateManager>,
    pub model_persistence_manager: Arc<PersiaPersistenceManager>,
    pub full_amount_manager: Arc<FullAmountManager>,
    pub shard_idx: usize,
}

impl HashMapShardedServiceInner {
    pub fn new(
        embedding: PersiaEmbeddingHolder,
        server_config: Arc<PersiaShardedServerConfig>,
        common_config: Arc<PersiaCommonConfig>,
        inc_update_manager: Arc<PerisaIncrementalUpdateManager>,
        model_persistence_manager: Arc<PersiaPersistenceManager>,
        full_amount_manager: Arc<FullAmountManager>,
        shard_idx: usize,
    ) -> Self {
        Self {
            embedding,
            optimizer: persia_futures::async_lock::RwLock::new(None),
            hyperparameter_config: persia_futures::async_lock::RwLock::new(None),
            hyperparameter_configured: persia_futures::async_lock::Mutex::new(false),
            server_config,
            common_config,
            inc_update_manager,
            model_persistence_manager,
            full_amount_manager,
            shard_idx,
        }
    }

    pub fn shard_idx(&self) -> usize {
        self.shard_idx
    }

    pub fn get_intent(&self) -> Result<PerisaIntent, ShardEmbeddingError> {
        let intent = self.common_config.intent.clone();
        Ok(intent)
    }

    pub async fn get_configuration(
        &self,
    ) -> Result<Arc<PersiaSparseModelHyperparameters>, ShardEmbeddingError> {
        let conf = self.hyperparameter_config.read().await;
        let conf = conf.as_ref();
        if let Some(conf) = conf {
            Ok(conf.clone())
        } else {
            Err(ShardEmbeddingError::NotConfiguredError)
        }
    }

    pub async fn batched_lookup(
        &self,
        req: Vec<(u64, usize)>,
    ) -> Result<Vec<f32>, ShardEmbeddingError> {
        let num_elements: usize = req.iter().map(|x| x.1).sum();
        let mut embeddings = Vec::with_capacity(num_elements);

        let mut index_miss_count: usize = 0;

        let intent = self.get_intent()?;
        let conf = match intent {
            PerisaIntent::Train => Some(self.get_configuration().await?),
            _ => None,
        };

        let optimizer = self.optimizer.read().await;

        tokio::task::block_in_place(|| match intent {
            PerisaIntent::Train => {
                if optimizer.is_none() {
                    return Err(ShardEmbeddingError::OptimizerNotFoundError);
                }
                let optimizer = optimizer.as_ref().unwrap();

                let mut evcited_ids = Vec::with_capacity(1000);

                req.iter().for_each(|(sign, dim)| {
                        let conf = conf.as_ref().unwrap();
                        let e = self.embedding.inner.get_refresh(sign);
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
                                    let (_, evcited) = self.embedding
                                        .inner
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
                                    let (_, evcited) = self
                                        .embedding
                                        .inner
                                        .insert(*sign, Arc::new(parking_lot::RwLock::new(entry)));

                                    if evcited.is_some() {
                                        evcited_ids.push(sign.clone());
                                    }
                                } else {
                                    embeddings.extend_from_slice(entry.read().emb());
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
            PerisaIntent::Eval => {
                req.iter().for_each(|(sign, dim)| {
                        let e = self.embedding.inner.get(sign);
                        match e {
                            None => {
                                embeddings.extend_from_slice(vec![0f32; *dim].as_slice());
                                index_miss_count += 1;
                            }
                            Some(entry) => {
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
                    });
                Ok(())
            }
            PerisaIntent::Infer(_) => {
                req.iter().for_each(|(sign, dim)| {
                        let e = self.embedding.inner.get(sign);
                        match e {
                            None => {
                                embeddings.extend_from_slice(vec![0f32; *dim].as_slice());
                                index_miss_count += 1;
                            }
                            Some(entry) => {
                                let entry_dim = { entry.read().dim_infer() };
                                if entry_dim != *dim {
                                    tracing::error!("dimensional mismatch on sign {}, in hashmap dim {}, requested dim {}",
                                        sign, entry_dim, dim);
                                    embeddings.extend_from_slice(vec![0f32; *dim].as_slice());
                                } else {
                                    embeddings.extend_from_slice(entry.read().emb_infer());
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
        let intent = self.common_config.intent.clone();
        match intent {
            PerisaIntent::Infer(_) => true,
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
    ) -> Result<(), ShardEmbeddingError> {
        let start_time = std::time::Instant::now();
        tokio::task::block_in_place(|| {
            let num_total_entry: usize = embeddings.len();
            let mut num_miss_entry: usize = 0;
            let mut num_recycled: usize = 0;
            for (id, entry) in embeddings.into_iter() {
                let cur = self.embedding.inner.get_refresh(&id);
                match cur {
                    Some(c) => {
                        let mut cur_embedding = c.write();
                        if !cur_embedding.copy_from_other(&entry) {
                            tracing::error!("set embedding to a wrong sign");
                        }
                    }
                    None => {
                        num_miss_entry = num_miss_entry + 1;
                        let pool = SET_EMBEDDING_POOL.get_or_init(|| EmbeddingPool::default());
                        let recycled = pool.pull(entry.dim_infer());

                        let entry_to_insert = {
                            match recycled {
                                Some(r) => {
                                    num_recycled = num_recycled + 1;
                                    r
                                }
                                None => {
                                    let new_entry =
                                        HashMapEmbeddingEntry::new_empty(entry.dim_infer());
                                    Arc::new(parking_lot::RwLock::new(new_entry))
                                }
                            }
                        };

                        {
                            let mut entry_to_insert_guard = entry_to_insert.write();
                            if !entry_to_insert_guard.copy_from_other(&entry) {
                                tracing::error!("set embedding to a wrong sign");
                            }
                        }

                        let (old, evicted) = self.embedding.inner.insert(id, entry_to_insert);
                        if let Some(o) = old {
                            let _pool_size = pool.push(o);
                        }
                        if let Some(e) = evicted {
                            let _pool_size = pool.push(e);
                        }
                    }
                }
            }

            if let Ok(m) = MetricsHolder::get() {
                m.set_embedding_time_cost
                    .observe(start_time.elapsed().as_secs_f64());
                let set_embedding_miss_ratio = num_miss_entry as f32 / num_total_entry as f32;
                m.set_embedding_miss_ratio
                    .set(set_embedding_miss_ratio.into());
                if num_miss_entry > 0 {
                    let set_embedding_recycled_ratio = num_recycled as f32 / num_miss_entry as f32;
                    m.set_embedding_recycled_ratio
                        .set(set_embedding_recycled_ratio.into());
                }
            }
        });

        Ok(())
    }

    pub async fn lookup_inference(&self, req: Bytes) -> Result<Bytes, ShardEmbeddingError> {
        let start_time = std::time::Instant::now();
        let indices =
            tokio::task::block_in_place(|| Vec::<(u64, usize)>::read_from_buffer(req.as_ref()));
        if indices.is_err() {
            return Err(ShardEmbeddingError::RpcError(
                "fail to des request".to_string(),
            ));
        }
        let indices = indices.unwrap();
        if let Ok(m) = MetricsHolder::get() {
            m.decode_indices_time_cost
                .observe(start_time.elapsed().as_secs_f64());
        }

        let embedding = self.batched_lookup(indices).await;
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
            Err(ShardEmbeddingError::RpcError(
                "fail to lookup embedding".to_string(),
            ))
        }
    }

    pub async fn lookup_mixed(
        &self,
        indices: Vec<(u64, usize)>,
    ) -> Result<Vec<f32>, ShardEmbeddingError> {
        let start_time = std::time::Instant::now();
        let embedding = self.batched_lookup(indices).await;
        if let Ok(m) = MetricsHolder::get() {
            m.lookup_mixed_batch_time_cost
                .observe(start_time.elapsed().as_secs_f64());
        }

        embedding
    }

    pub async fn update_gradient_mixed(
        &self,
        req: (Vec<u64>, Vec<f32>),
    ) -> Result<(), ShardEmbeddingError> {
        let conf = self.get_configuration().await?;
        let (signs, remaining_gradients) = req;
        let mut remaining_gradients = remaining_gradients.as_slice();
        let mut indices_to_commit = Vec::with_capacity(signs.len());
        let mut gradient_id_miss_count = 0;

        let optimizer = self.optimizer.read().await;
        if optimizer.is_none() {
            return Err(ShardEmbeddingError::OptimizerNotFoundError);
        }

        let optimizer = optimizer.as_ref().unwrap();

        tokio::task::block_in_place(|| {
            for sign in signs {
                if let Some(entry) = self.embedding.inner.get(&sign) {
                    let entry_dim = { entry.read().dim() };
                    let (grad, r) = remaining_gradients.split_at(entry_dim);
                    remaining_gradients = r;

                    {
                        let mut entry = entry.write();
                        let emb_entry_slice = entry.as_mut_emb_entry_slice();
                        optimizer.update(emb_entry_slice, grad, entry_dim);

                        if conf.enable_weight_bound {
                            unsafe {
                                persia_simd::weight_bound(
                                    &mut emb_entry_slice[..entry_dim],
                                    conf.weight_bound,
                                );
                            }
                        }
                    }

                    indices_to_commit.push((sign, entry.clone()));
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
        optimizer: Optimizer,
    ) -> Result<(), ShardEmbeddingError> {
        {
            let mut optimizer_ = self.optimizer.write().await;
            *optimizer_ = Some(Arc::new(optimizer.to_optimizable()));
        }
        Ok(())
    }

    pub async fn configure(
        &self,
        config: PersiaSparseModelHyperparameters,
    ) -> Result<(), ShardEmbeddingError> {
        {
            let mut conf_guard = self.hyperparameter_config.write().await;
            *conf_guard = Some(Arc::new(config));
        }
        *self.hyperparameter_configured.lock().await = true;
        tracing::info!("sharded server configured");
        Ok(())
    }

    pub async fn dump(&self, dir: String) -> Result<(), ShardEmbeddingError> {
        let dst_dir = PathBuf::from(dir);
        self.model_persistence_manager
            .dump_full_amount_embedding(dst_dir)?;
        Ok(())
    }

    pub async fn load(&self, dir: String) -> Result<(), ShardEmbeddingError> {
        let dst_dir = PathBuf::from(dir);
        self.model_persistence_manager
            .load_embedding_from_dir(dst_dir)?;
        Ok(())
    }

    pub async fn get_address(&self) -> Result<String, ShardEmbeddingError> {
        let instance_info = InstanceInfo::get()?;
        let address = format!("{}:{}", instance_info.ip_address, instance_info.port);
        Ok(address)
    }

    pub async fn get_replica_info(&self) -> Result<PersiaReplicaInfo, ShardEmbeddingError> {
        let replica_info = PersiaReplicaInfo::get()?;
        let replica_info = replica_info.as_ref().clone();
        Ok(replica_info)
    }
}

#[derive(Clone)]
pub struct HashMapShardedService {
    pub inner: Arc<HashMapShardedServiceInner>,
    pub shutdown_channel:
        Arc<persia_futures::async_lock::RwLock<Option<tokio::sync::oneshot::Sender<()>>>>,
}

#[persia_rpc::service]
impl HashMapShardedService {
    pub async fn ready_for_serving(&self, _req: ()) -> bool {
        self.inner.ready_for_serving().await
    }

    pub async fn model_manager_status(&self, _req: ()) -> PersiaPersistenceStatus {
        self.inner.model_manager_status().await
    }

    pub async fn set_embedding(
        &self,
        req: Vec<(u64, HashMapEmbeddingEntry)>,
    ) -> Result<(), ShardEmbeddingError> {
        self.inner.set_embedding(req).await
    }

    pub async fn lookup_inference(&self, req: Bytes) -> Result<Bytes, ShardEmbeddingError> {
        self.inner.lookup_inference(req).await
    }

    pub async fn lookup_mixed(
        &self,
        req: Vec<(u64, usize)>,
    ) -> Result<Vec<f32>, ShardEmbeddingError> {
        self.inner.lookup_mixed(req).await
    }

    pub async fn shard_idx(&self, _req: ()) -> usize {
        self.inner.shard_idx()
    }

    pub async fn update_gradient_mixed(
        &self,
        req: (Vec<u64>, Vec<f32>),
    ) -> Result<(), ShardEmbeddingError> {
        self.inner.update_gradient_mixed(req).await
    }

    pub async fn configure(
        &self,
        config: PersiaSparseModelHyperparameters,
    ) -> Result<(), ShardEmbeddingError> {
        self.inner.configure(config).await
    }

    pub async fn dump(&self, req: String) -> Result<(), ShardEmbeddingError> {
        self.inner.dump(req).await
    }

    pub async fn load(&self, req: String) -> Result<(), ShardEmbeddingError> {
        self.inner.load(req).await
    }

    pub async fn register_optimizer(
        &self,
        optimizer: Optimizer,
    ) -> Result<(), ShardEmbeddingError> {
        self.inner.register_optimizer(optimizer).await
    }

    pub async fn shutdown(&self, _req: ()) -> Result<(), ShardEmbeddingError> {
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
                    Err(ShardEmbeddingError::ShutdownError)
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
pub struct EmbeddingServerNatsStub {
    pub inner: Arc<HashMapShardedServiceInner>,
}

#[persia_nats_marcos::stub]
impl EmbeddingServerNatsStub {
    pub async fn ready_for_serving(&self, _req: ()) -> bool {
        self.inner.ready_for_serving().await
    }

    pub async fn model_manager_status(&self, _req: ()) -> PersiaPersistenceStatus {
        self.inner.model_manager_status().await
    }

    pub async fn shard_idx(&self, _req: ()) -> usize {
        self.inner.shard_idx()
    }

    pub async fn configure(
        &self,
        config: PersiaSparseModelHyperparameters,
    ) -> Result<(), ShardEmbeddingError> {
        self.inner.configure(config).await
    }

    pub async fn dump(&self, req: String) -> Result<(), ShardEmbeddingError> {
        self.inner.dump(req).await
    }

    pub async fn load(&self, req: String) -> Result<(), ShardEmbeddingError> {
        self.inner.load(req).await
    }

    pub async fn get_address(&self, _req: ()) -> Result<String, ShardEmbeddingError> {
        self.inner.get_address().await
    }

    pub async fn get_replica_info(
        &self,
        _req: (),
    ) -> Result<PersiaReplicaInfo, ShardEmbeddingError> {
        self.inner.get_replica_info().await
    }

    pub async fn register_optimizer(
        &self,
        optimizer: Optimizer,
    ) -> Result<(), ShardEmbeddingError> {
        self.inner.register_optimizer(optimizer).await
    }
}
