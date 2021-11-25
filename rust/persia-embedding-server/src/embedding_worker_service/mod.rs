use crate::embedding_parameter_service::{
    EmbeddingParameterNatsServicePublisher, EmbeddingParameterServerError,
    EmbeddingParameterServiceClient,
};

use std::iter::FromIterator;
use std::ops::MulAssign;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use persia_libs::{
    async_lock::RwLock,
    backoff::{future::retry, ExponentialBackoff},
    bytes, futures,
    hashbrown::HashMap,
    hyper,
    itertools::Itertools,
    lz4, ndarray, once_cell,
    smol::block_on,
    thiserror, tokio, tracing,
};
use snafu::ResultExt;

use persia_common::{
    grad::{EmbeddingGradientBatch, Gradients, SkippableFeatureEmbeddingGradientBatch},
    ndarray_f16_to_f32, ndarray_f32_to_f16,
    optim::OptimizerConfig,
    EmbeddingBatch, FeatureEmbeddingBatch, FeatureRawEmbeddingBatch, FeatureSumEmbeddingBatch,
    IDTypeFeatureBatch, IDTypeFeatureRemoteRef, SingleSignInFeatureBatch,
};
use persia_embedding_config::{
    EmbeddingConfig, EmbeddingWorkerConfig, InstanceInfo, PersiaCommonConfig,
    PersiaEmbeddingModelHyperparameters, PersiaGlobalConfigError, PersiaReplicaInfo, SlotConfig,
};
use persia_embedding_holder::emb_entry::HashMapEmbeddingEntry;
use persia_metrics::{
    Gauge, GaugeVec, IntCounterVec, PersiaMetricsManager, PersiaMetricsManagerError,
};
use persia_model_manager::{
    EmbeddingModelManager, EmbeddingModelManagerError, EmbeddingModelManagerStatus,
};
use persia_nats_client::{NatsClient, NatsError};
use persia_speedy::{Readable, Writable};

static METRICS_HOLDER: once_cell::sync::OnceCell<MetricsHolder> = once_cell::sync::OnceCell::new();

struct MetricsHolder {
    pub batch_unique_indices_rate: GaugeVec,
    pub num_pending_batches: Gauge,
    pub staleness: Gauge,
    pub nan_count: IntCounterVec,
    pub nan_grad_skipped: IntCounterVec,
    pub lookup_create_requests_time_cost_sec: Gauge,
    pub lookup_rpc_time_cost_sec: Gauge,
    pub update_gradient_time_cost_sec: Gauge,
    pub summation_time_cost_sec: Gauge,
    pub lookup_batched_time_cost_sec: Gauge,
}

impl MetricsHolder {
    pub fn get() -> Result<&'static Self, PersiaMetricsManagerError> {
        METRICS_HOLDER.get_or_try_init(|| {
            let m = PersiaMetricsManager::get()?;
            let holder = Self {
                batch_unique_indices_rate: m
                    .create_gauge_vec("batch_unique_indices_rate", "unique indices rate in a batch for different features")?,
                num_pending_batches: m.create_gauge(
                    "num_pending_batches", 
                    "num batches already sent to embedding worker but still waiting for nn worker to trigger lookup.
                    The pending batches will stored in forward buffer, which capacity is configurable by global_config.yaml. 
                    Once the buffer full, embedding worker may not accept new batches."
                )?,
                staleness: m.create_gauge(
                    "staleness", 
                    "staleness of embedding model. The iteration of dense model run one by one, while the embedding lookup happened 
                    before concurrently. The staleness describe the delay of embeddings. The value of staleness start with 0, 
                    increase one when lookup a batch, decrease one when a batch update its gradients"
                )?,
                nan_count: m.create_counter_vec("nan_count","nan count of gradient pushed to emb server")?,
                nan_grad_skipped: m.create_counter_vec("nan_grad_skipped","nan count of gradient filtered by gpu node")?,
                lookup_create_requests_time_cost_sec: m.create_gauge(
                    "lookup_create_requests_time_cost_sec", 
                    "lookup preprocess time cost on embedding worker. Include ID hashing, dividing id accroding feature groups and embedding servers."
                )?,
                lookup_rpc_time_cost_sec: m.create_gauge(
                    "lookup_rpc_time_cost_sec", 
                    "lookup embedding time cost on embedding worker for a batch, include lookup on embedding server and network transmission."
                )?,
                update_gradient_time_cost_sec: m
                    .create_gauge("update_gradient_time_cost_sec", "update gradient time cost on embedding worker for a batch.")?,
                summation_time_cost_sec: m.create_gauge(
                    "summation_time_cost_sec",
                     "lookup postprocess time cost on embedding worker, mainly is embedding summation."
                )?,
                lookup_batched_time_cost_sec: m.create_gauge(
                    "lookup_batched_time_cost_sec",
                    "lookup and pre/post process time cost on embedding worker."
                )?,
            };
            Ok(holder)
        })
    }
}

#[derive(thiserror::Error, Debug, Readable, Writable)]
pub enum EmbeddingWorkerError {
    #[error("rpc error")]
    RpcError(String),
    #[error("embedding server error: {0}")]
    EmbeddingParameterServerError(#[from] EmbeddingParameterServerError),
    #[error("embedding model manager error: {0}")]
    EmbeddingModelManagerError(#[from] EmbeddingModelManagerError),
    #[error("forward id not found")]
    ForwardIdNotFound(u64),
    #[error("forward failed")]
    ForwardFailed(String),
    #[error("backward failed")]
    BackwardFailed(String),
    #[error("connection failed")]
    ConnectionError,
    #[error("lookup server ip address error")]
    LookupIpAddressError,
    #[error("server replica idx not match error")]
    ReplicaIdxNotMatchError,
    #[error("gradient contains nan")]
    NanGradient,
    #[error("nats error: {0}")]
    NatsError(#[from] NatsError),
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("forward buffer full")]
    ForwardBufferFull,
    #[error("data src idx not set")]
    DataSrcIdxNotSet,
}

pub struct AllEmbeddingServerClient {
    clients: RwLock<Vec<Arc<EmbeddingParameterServiceClient>>>,
    nats_publisher: Option<EmbeddingParameterNatsServicePublisher>,
    dst_replica_size: usize,
}

impl AllEmbeddingServerClient {
    pub async fn with_nats(nats_publisher: EmbeddingParameterNatsServicePublisher) -> Self {
        tracing::info!("trying to get replica info of embedding servers");

        let mut backoff = ExponentialBackoff::default();
        backoff.max_interval = std::time::Duration::from_millis(500);

        let dst_replica_info: Result<PersiaReplicaInfo, _> = retry(backoff, || async {
            let resp = AllEmbeddingServerClient::get_dst_replica_info(&nats_publisher).await;
            if resp.is_err() {
                tracing::warn!(
                    "failed to get replica info of embedding servers, due to {:?}, retrying",
                    resp
                );
            } else {
                tracing::info!(
                    "succeed to get replica info of embedding servers, {:?}",
                    resp
                );
            }
            Ok(resp?)
        })
        .await;

        let dst_replica_info =
            dst_replica_info.expect("failed to get replica info of embedding servers");

        let instance = Self {
            clients: RwLock::new(Vec::with_capacity(dst_replica_info.replica_size)),
            nats_publisher: Some(nats_publisher),
            dst_replica_size: dst_replica_info.replica_size,
        };

        let servers = instance
            .get_all_addresses()
            .await
            .expect("failed to get ips of embedding servers");

        instance
            .update_rpc_clients(servers, false)
            .await
            .expect("failed to init rpc client for embedding servers");

        instance
    }

    pub async fn with_addrs(servers: Vec<String>) -> Self {
        tracing::info!(
            "AllEmbeddingServerClient::with_addrs, embedding servers are {:?}",
            servers
        );
        let instance = Self {
            clients: RwLock::new(Vec::with_capacity(servers.len())),
            nats_publisher: None,
            dst_replica_size: servers.len(),
        };

        instance
            .update_rpc_clients(servers, true)
            .await
            .expect("failed to init rpc client for embedding server");

        instance
    }

    pub async fn ready_for_serving(&self) -> bool {
        let futs = (0..self.replica_size()).map(|client_idx| async move {
            let client = self.get_client_by_index(client_idx).await;
            let resp = client.ready_for_serving(&()).await;
            if let Ok(x) = resp {
                if x {
                    return Ok(());
                }
            }
            return Err(());
        });

        futures::future::try_join_all(futs).await.is_ok()
    }

    pub async fn model_manager_status(&self) -> Vec<EmbeddingModelManagerStatus> {
        let futs = (0..self.replica_size()).map(|client_idx| async move {
            let client = self.get_client_by_index(client_idx).await;
            client.model_manager_status(&()).await
        });

        let status: Vec<_> = futures::future::try_join_all(futs).await.unwrap_or(vec![
                EmbeddingModelManagerStatus::Failed(EmbeddingModelManagerError::FailedToGetStatus);
                self.replica_size()
            ]);

        return status;
    }

    pub fn replica_size(&self) -> usize {
        self.dst_replica_size
    }

    pub async fn get_client_by_index(
        &self,
        client_index: usize,
    ) -> Arc<EmbeddingParameterServiceClient> {
        let clients = self.clients.read().await;
        clients.get(client_index).unwrap().clone()
    }

    pub async fn get_dst_replica_info(
        nats_publisher: &EmbeddingParameterNatsServicePublisher,
    ) -> Result<PersiaReplicaInfo, EmbeddingWorkerError> {
        let dst_replica_info = nats_publisher.publish_get_replica_info(&(), None).await??;
        Ok(dst_replica_info)
    }

    pub async fn get_address(&self, replica_index: usize) -> Result<String, EmbeddingWorkerError> {
        let addr = self
            .nats_publisher
            .as_ref()
            .expect("nats_publisher is None, you are using inference mode")
            .publish_get_address(&(), Some(replica_index))
            .await??;
        Ok(addr)
    }

    pub async fn get_all_addresses(&self) -> Result<Vec<String>, EmbeddingWorkerError> {
        let futs = (0..self.dst_replica_size).map(|replica_index| async move {
            tracing::info!(
                "trying to get ip address of embedding server {}",
                replica_index
            );
            let mut backoff = ExponentialBackoff::default();
            backoff.max_interval = std::time::Duration::from_millis(500);
            retry(backoff, || async {
                let addr = self.get_address(replica_index).await;
                if addr.is_err() {
                    tracing::warn!(
                        "failed to get address of embedding servers {}, due to {:?}, retrying",
                        replica_index,
                        addr
                    );
                } else {
                    tracing::info!(
                        "succeed to get address of embedding servers {}, {:?}",
                        replica_index,
                        addr
                    );
                }
                Ok(addr?)
            })
            .await
        });

        let servers: Vec<_> = futures::future::try_join_all(futs).await?;
        Ok(servers)
    }

    pub async fn update_rpc_clients(
        &self,
        servers: Vec<String>,
        ready_for_serving: bool,
    ) -> Result<(), EmbeddingWorkerError> {
        let mut clients = self.clients.write().await;
        clients.clear();

        for server_addr in servers {
            let rpc_client = persia_rpc::RpcClient::new(server_addr.as_str()).unwrap();
            let client = EmbeddingParameterServiceClient::new(rpc_client);
            clients.push(Arc::new(client));
        }

        for (i, c) in clients.iter().enumerate() {
            while ready_for_serving && !c.ready_for_serving(&()).await.unwrap_or(false) {
                tracing::info!("waiting for embedding server ready...");
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }

            tracing::info!("start to call replica_index");
            match c.replica_index(&()).await {
                Ok(idx) => {
                    if idx != i {
                        tracing::error!("replica index wrong");
                        return Err(EmbeddingWorkerError::ReplicaIdxNotMatchError);
                    } else {
                        tracing::info!("succeed to call replica_index");
                    }
                }
                Err(e) => {
                    tracing::error!("failed to call replica_index due to {:?}", e);
                    return Err(EmbeddingWorkerError::ConnectionError);
                }
            }
        }

        Ok(())
    }
}

#[inline]
pub fn sign_to_shard_modulo(sign: u64, replica_size: u64) -> u64 {
    let sign = farmhash::hash64(&sign.to_le_bytes());
    sign % replica_size
}

#[inline]
pub fn indices_to_hashstack_indices(
    indices: &mut IDTypeFeatureBatch,
    config: &EmbeddingConfig,
) -> () {
    for feature_batch in indices.batches.iter_mut() {
        let slot_conf = config.get_slot_by_feature_name(&feature_batch.feature_name);
        if slot_conf.hash_stack_config.hash_stack_rounds > 0 {
            let mut hash_stack_indices: Vec<HashMap<u64, Vec<(u16, u16)>>> =
                vec![HashMap::new(); slot_conf.hash_stack_config.hash_stack_rounds];
            let mut hashed2index_batch_idx: HashMap<u64, i64> = HashMap::with_capacity(
                feature_batch.index_batch.len() * slot_conf.hash_stack_config.hash_stack_rounds,
            );
            feature_batch.index_batch.iter().enumerate().for_each(
                |(distinct_tensor_idx, single_sign)| {
                    let mut hashed_sign = single_sign.sign;
                    for (round, map) in hash_stack_indices.iter_mut().enumerate() {
                        hashed_sign = farmhash::hash64(&hashed_sign.to_le_bytes());
                        let hashed_sign_bucket = hashed_sign.clone()
                            % slot_conf.hash_stack_config.embedding_size as u64
                            + (round * slot_conf.hash_stack_config.embedding_size) as u64;
                        // TODO: to avoid hash conflict, try replace hashed2index_batch_idx to key2list
                        hashed2index_batch_idx
                            .insert(hashed_sign_bucket, distinct_tensor_idx as i64);
                        map.entry(hashed_sign_bucket)
                            .or_insert_with(|| {
                                Vec::with_capacity(single_sign.in_which_batch_samples.len())
                            })
                            .extend_from_slice(single_sign.in_which_batch_samples.as_slice());
                    }
                },
            );
            let mut hashed_index_batch: Vec<SingleSignInFeatureBatch> = Vec::with_capacity(
                hash_stack_indices.first().unwrap().len()
                    * slot_conf.hash_stack_config.hash_stack_rounds,
            );
            for map in hash_stack_indices.into_iter() {
                for (k, v) in map {
                    hashed_index_batch.push(SingleSignInFeatureBatch {
                        sign: k,
                        in_which_batch_samples: v,
                    });
                }
            }
            feature_batch.index_batch = hashed_index_batch;
            feature_batch.hashed2index_batch_idx = hashed2index_batch_idx;
            feature_batch.sample_num_signs = feature_batch
                .sample_num_signs
                .iter()
                .map(|x| x * slot_conf.hash_stack_config.hash_stack_rounds as u32)
                .collect();
        }
    }
}

#[inline]
pub fn indices_add_prefix(indices: &mut IDTypeFeatureBatch, config: &EmbeddingConfig) -> () {
    let feature_spacing = if config.feature_index_prefix_bit > 0 {
        (1u64 << (u64::BITS - config.feature_index_prefix_bit as u32)) - 1
    } else {
        u64::MAX
    };
    for feature_batch in indices.batches.iter_mut() {
        let slot_conf = config.get_slot_by_feature_name(&feature_batch.feature_name);
        if slot_conf.index_prefix > 0 {
            for single_sign in feature_batch.index_batch.iter_mut() {
                single_sign.sign %= feature_spacing;
                single_sign.sign += slot_conf.index_prefix;
            }
            let mut index_prefix_mapping: HashMap<u64, i64> =
                HashMap::with_capacity(feature_batch.hashed2index_batch_idx.len());

            feature_batch
                .hashed2index_batch_idx
                .iter()
                .for_each(|(id, batch_idx)| {
                    index_prefix_mapping
                        .insert(id % feature_spacing + slot_conf.index_prefix, *batch_idx);
                });
            feature_batch.hashed2index_batch_idx = index_prefix_mapping;
        }
    }
}

#[derive(Clone)]
pub struct SignWithConfig {
    sign: u64,
    sign_idx: usize,
    feature_idx: usize,
    dim: usize,
}

impl SignWithConfig {
    pub fn get_sign(&self) -> u64 {
        self.sign
    }
    pub fn get_dim(&self) -> usize {
        self.dim
    }
}

pub fn lookup_batched_all_slots_preprocess(
    indices: &mut IDTypeFeatureBatch,
    config: &EmbeddingConfig,
    replica_size: u64,
) -> Vec<Vec<SignWithConfig>> {
    #[inline]
    fn indices_to_sharded_indices(
        indices: &IDTypeFeatureBatch,
        config: &EmbeddingConfig,
        replica_size: u64,
    ) -> Vec<Vec<SignWithConfig>> {
        // TODO: optimization point: duplicate sign may exists in lookup result, split
        // Vec<Vec<SignWithConfig>> into Vec<Vec<id>>,
        let mut results = vec![Vec::new(); replica_size as usize];
        for (feature_idx, feature_batch) in indices.batches.iter().enumerate() {
            let slot_conf = config.get_slot_by_feature_name(&feature_batch.feature_name);
            for (sign_idx, single_sign) in feature_batch.index_batch.iter().enumerate() {
                let replica_index = sign_to_shard_modulo(single_sign.sign, replica_size);
                unsafe {
                    results
                        .get_unchecked_mut(replica_index as usize)
                        .push(SignWithConfig {
                            sign: single_sign.sign,
                            sign_idx,
                            feature_idx,
                            dim: slot_conf.dim,
                        });
                }
            }
        }
        results
    }

    indices_to_hashstack_indices(indices, config);
    indices_add_prefix(indices, config);
    indices_to_sharded_indices(&indices, config, replica_size)
}

pub fn lookup_batched_all_slots_postprocess<'a>(
    indices: &IDTypeFeatureBatch,
    forwarded_groups: Vec<(Vec<f32>, Vec<SignWithConfig>)>,
    config: &'a EmbeddingConfig,
) -> Vec<FeatureEmbeddingBatch> {
    struct LookupResultWithSlotConfig<'a> {
        result: ndarray::Array2<f32>,
        config: &'a SlotConfig,
        sign2idx: HashMap<u64, i64>,
    }

    let mut results: Vec<LookupResultWithSlotConfig<'a>> = indices
        .batches
        .iter()
        .map(|x| {
            let slot_conf = config.get_slot_by_feature_name(&x.feature_name);
            let (feature_len, sign2idx) = if slot_conf.embedding_summation {
                (x.batch_size as usize, HashMap::new())
            } else {
                let distinct_id_size = if slot_conf.hash_stack_config.hash_stack_rounds > 0 {
                    x.index_batch.len() / slot_conf.hash_stack_config.hash_stack_rounds
                } else {
                    x.index_batch.len()
                };
                (distinct_id_size + 1, x.hashed2index_batch_idx.clone())
            };
            LookupResultWithSlotConfig {
                result: ndarray::Array2::<f32>::zeros((feature_len, slot_conf.dim)),
                config: slot_conf,
                sign2idx,
            }
        })
        .collect_vec();

    for group in forwarded_groups {
        let (embeddings, signs) = group;
        let lookup_raw_results = {
            let mut results = Vec::with_capacity(signs.len());
            let mut embeddings_slice = embeddings.as_slice();
            for sign in &signs {
                let (l, r) = embeddings_slice.split_at(sign.dim);
                embeddings_slice = r;
                results.push(l);
            }
            assert_eq!(
                embeddings_slice.len(),
                0,
                "embeddings lookup results do not match dimension"
            );
            results
        };
        for (emb, single_sign) in lookup_raw_results.iter().zip(signs) {
            let feature_idx = single_sign.feature_idx;
            let result = unsafe { results.get_unchecked_mut(feature_idx) };
            if !result.config.embedding_summation {
                let mut row = result
                    .result
                    .row_mut(*result.sign2idx.get(&single_sign.sign).unwrap() as usize + 1);
                let row = row.as_slice_mut().unwrap();
                row.clone_from_slice(emb);
            } else {
                let sign_idx = single_sign.sign_idx;
                let single_sign = unsafe {
                    indices
                        .batches
                        .get_unchecked(feature_idx)
                        .index_batch
                        .get_unchecked(sign_idx)
                };
                for (batch_id, _) in &single_sign.in_which_batch_samples {
                    let mut row = result.result.row_mut(*batch_id as usize);
                    let row = row.as_slice_mut().unwrap();
                    unsafe {
                        persia_simd::add_assign_avx2(row, emb);
                    }
                }
            }
        }
    }

    let batches = results
        .into_iter()
        .zip(indices.batches.iter())
        .map(|(mut x, indices)| {
            if x.config.embedding_summation {
                if x.config.sqrt_scaling {
                    let sample_num_ids = ndarray::Array2::from_shape_vec(
                        (indices.sample_num_signs.len(), 1),
                        indices.sample_num_signs.clone(),
                    )
                    .unwrap();
                    x.result.mul_assign(
                        &sample_num_ids.mapv(|x| (std::cmp::max(x, 1) as f32).sqrt().recip()),
                    );
                }
                FeatureEmbeddingBatch::SumEmbedding(FeatureSumEmbeddingBatch {
                    feature_name: indices.feature_name.clone(),
                    embeddings: ndarray_f32_to_f16(&x.result),
                })
            } else {
                if x.config.sqrt_scaling && x.config.hash_stack_config.hash_stack_rounds > 1 {
                    x.result.mul_assign(
                        (x.config.hash_stack_config.hash_stack_rounds as f32)
                            .sqrt()
                            .recip(),
                    )
                }
                // transform distinct_id tensor to origin batch format
                let mut index: Vec<i64> =
                    vec![0; indices.batch_size as usize * x.config.sample_fixed_size];
                let mut sample_id_num: Vec<usize> = vec![0; indices.batch_size as usize];
                let mut transform_id_set =
                    std::collections::HashSet::with_capacity(indices.index_batch.len());
                let id2idx = &x.sign2idx;

                indices.index_batch.iter().for_each(|item| {
                    let distinct_tensor_idx = id2idx.get(&item.sign).unwrap();
                    if !transform_id_set.contains(distinct_tensor_idx) {
                        transform_id_set.insert(distinct_tensor_idx);
                        for (batch_idx, col_idx) in &item.in_which_batch_samples {
                            let batch_idx = *batch_idx as usize;
                            let col_idx = *col_idx as usize;
                            if sample_id_num[batch_idx] < x.config.sample_fixed_size
                                && col_idx < x.config.sample_fixed_size
                            {
                                index[batch_idx * x.config.sample_fixed_size + col_idx] =
                                    distinct_tensor_idx + 1;
                                sample_id_num[batch_idx] += 1;
                            }
                        }
                    }
                });
                FeatureEmbeddingBatch::RawEmbedding(FeatureRawEmbeddingBatch {
                    feature_name: indices.feature_name.clone(),
                    embeddings: ndarray_f32_to_f16(&x.result),
                    index,
                    sample_id_num,
                })
            }
        })
        .collect();

    batches
}

#[repr(align(64))] // cache line optimization
pub struct EmbeddingWorkerInner {
    pub all_embedding_server_client: AllEmbeddingServerClient,
    pub replica_size: u64,
    pub forward_id: AtomicU64,
    pub cannot_forward_batched_time: crossbeam::atomic::AtomicCell<SystemTime>, // TODO: use parking_lot::RwLock to replace
    pub forward_id_buffer:
        persia_libs::async_lock::RwLock<HashMap<usize, HashMap<u64, IDTypeFeatureBatch>>>,
    pub post_forward_buffer: persia_libs::async_lock::RwLock<HashMap<u64, IDTypeFeatureBatch>>,
    pub staleness: AtomicUsize,
    pub embedding_config: Arc<EmbeddingConfig>,
    pub embedding_worker_config: Arc<EmbeddingWorkerConfig>,
    pub embedding_model_manager: Arc<EmbeddingModelManager>,
}

impl EmbeddingWorkerInner {
    fn get_id(&self) -> u64 {
        self.forward_id.fetch_add(1, Ordering::AcqRel)
    }

    fn is_master_server(&self) -> Result<bool, EmbeddingWorkerError> {
        let repilca_info = PersiaReplicaInfo::get()?;
        Ok(repilca_info.is_master())
    }

    async fn forward_batched(
        &self,
        indices: IDTypeFeatureBatch,
        batcher_idx: usize,
    ) -> Result<u64, EmbeddingWorkerError> {
        let id = self.get_id();

        if let Ok(m) = MetricsHolder::get() {
            for batch in indices.batches.iter() {
                let num_ids_this_batch: usize = tokio::task::block_in_place(|| {
                    batch.sample_num_signs.iter().sum::<u32>() as usize
                });
                let num_unique_ids: usize = batch.index_batch.len();
                let batch_unique_indices_rate = num_unique_ids as f32 / num_ids_this_batch as f32;

                m.batch_unique_indices_rate
                    .with_label_values(&[batch.feature_name.as_str()])
                    .set(batch_unique_indices_rate.into());

                m.num_pending_batches
                    .set(self.forward_id_buffer.read().await.len() as f64);
                m.staleness
                    .set(self.staleness.load(Ordering::Acquire) as f64);
            }
        }

        {
            let mut indices = indices;
            indices.enter_forward_id_buffer_time = Some(SystemTime::now());

            let mut forward_id_buffer = self.forward_id_buffer.write().await;
            let sub_buffer = forward_id_buffer.get_mut(&batcher_idx);
            match sub_buffer {
                Some(b) => {
                    b.insert(id, indices);
                }
                None => {
                    let mut new_sub =
                        HashMap::with_capacity(self.embedding_worker_config.forward_buffer_size);
                    new_sub.insert(id, indices);
                    forward_id_buffer.insert(batcher_idx, new_sub);
                }
            }
        }
        Ok(id)
    }

    pub async fn update_all_batched_gradients(
        &self,
        embedding_gradient_batch: &mut EmbeddingGradientBatch,
        indices: IDTypeFeatureBatch,
    ) -> Result<(), EmbeddingWorkerError> {
        let start_time = std::time::Instant::now();

        let indices_kv: HashMap<_, _> = indices
            .batches
            .iter()
            .map(|batch| (batch.feature_name.as_str(), batch))
            .collect();

        let mut sharded_gradients = vec![vec![]; self.all_embedding_server_client.replica_size()];
        let mut sharded_gradient_signs =
            vec![vec![]; self.all_embedding_server_client.replica_size()];

        for gradient in embedding_gradient_batch.gradients.iter_mut() {
            match gradient {
                SkippableFeatureEmbeddingGradientBatch::GradientBatch(feature_gradient) => {
                    let feature_batch = indices_kv
                        .get(&feature_gradient.feature_name.as_str())
                        .unwrap();
                    let slot_conf = self
                        .embedding_config
                        .get_slot_by_feature_name(feature_batch.feature_name.as_str());
                    let raw_gradients = std::mem::take(&mut feature_gradient.gradients);

                    if tokio::task::block_in_place(|| match &raw_gradients {
                        Gradients::F16(f16_gradients) => {
                            f16_gradients.as_slice().unwrap().iter().any(|x| x.is_nan())
                        }
                        Gradients::F32(f32_gradients) => {
                            f32_gradients.as_slice().unwrap().iter().any(|x| x.is_nan())
                        }
                    }) {
                        tracing::warn!("nan found in gradient update, skipping");
                        if let Ok(m) = MetricsHolder::get() {
                            m.nan_count
                                .with_label_values(&[feature_batch.feature_name.as_str()])
                                .inc();
                        }
                        continue;
                    }
                    let mut f32_gradients = tokio::task::block_in_place(|| match raw_gradients {
                        Gradients::F16(gradients_array) => ndarray_f16_to_f32(&gradients_array),
                        Gradients::F32(gradients_array) => gradients_array,
                    });
                    if (feature_gradient.scale_factor - 1.0).abs() > f32::EPSILON {
                        let scale = feature_gradient.scale_factor.recip();
                        assert!(scale.is_finite(), "scale on gradient must be finite");
                        tokio::task::block_in_place(|| f32_gradients.mul_assign(scale));
                    }

                    if slot_conf.sqrt_scaling {
                        tokio::task::block_in_place(|| {
                            if slot_conf.embedding_summation {
                                let sample_num_ids = ndarray::Array2::from_shape_vec(
                                    (feature_batch.sample_num_signs.len(), 1),
                                    feature_batch.sample_num_signs.clone(),
                                )
                                .unwrap();
                                f32_gradients.mul_assign(
                                    &sample_num_ids.mapv(|x| (x as f32).sqrt().recip()),
                                );
                            } else {
                                if slot_conf.hash_stack_config.hash_stack_rounds > 0 {
                                    f32_gradients.mul_assign(
                                        (slot_conf.hash_stack_config.hash_stack_rounds as f32)
                                            .sqrt()
                                            .recip(),
                                    );
                                }
                            }
                        });
                    }

                    tokio::task::block_in_place(|| {
                        let mut sign_gradients = ndarray::Array2::<f32>::zeros((
                            feature_batch.index_batch.len(),
                            slot_conf.dim,
                        ));
                        let hashed2index_batch_idx = &feature_batch.hashed2index_batch_idx;
                        for (row, single_sign) in feature_batch.index_batch.iter().enumerate() {
                            let mut sign_grad = sign_gradients.row_mut(row);
                            let sign_grad = sign_grad.as_slice_mut().unwrap();

                            if !slot_conf.embedding_summation {
                                let batch_idx =
                                    hashed2index_batch_idx.get(&single_sign.sign).unwrap();
                                unsafe {
                                    persia_simd::add_assign_avx2(
                                        sign_grad,
                                        f32_gradients.row(*batch_idx as usize).as_slice().unwrap(),
                                    );
                                }
                            } else {
                                single_sign.in_which_batch_samples.iter().for_each(
                                    |(batch_id, _)| {
                                        let row_grad = f32_gradients.row(*batch_id as usize);
                                        unsafe {
                                            persia_simd::add_assign_avx2(
                                                sign_grad,
                                                row_grad.as_slice().unwrap(),
                                            );
                                        }
                                    },
                                );
                            }
                        }
                        for (grad, sign) in sign_gradients
                            .axis_iter(ndarray::Axis(0))
                            .zip(feature_batch.index_batch.iter())
                        {
                            let replica_index = sign_to_shard_modulo(sign.sign, self.replica_size);
                            sharded_gradients[replica_index as usize]
                                .extend_from_slice(grad.as_slice().unwrap());
                            sharded_gradient_signs[replica_index as usize].push(sign.sign);
                        }
                    });
                }
                SkippableFeatureEmbeddingGradientBatch::Skipped(skipped) => {
                    if let Ok(m) = MetricsHolder::get() {
                        m.nan_grad_skipped
                            .with_label_values(&[skipped.feature_name.as_str()])
                            .inc();
                    }
                    continue;
                }
            }
        }

        let futs = sharded_gradients
            .into_iter()
            .zip(sharded_gradient_signs)
            .enumerate()
            .map(|(replica_index, (grads, signs))| {
                let client = block_on(
                    self.all_embedding_server_client
                        .get_client_by_index(replica_index),
                );
                async move {
                    let start_time = Instant::now();
                    client
                        .update_gradient_mixed(&(signs, grads))
                        .await
                        .map_err(|e| EmbeddingWorkerError::RpcError(format!("{:?}", e)))??;
                    let result = Ok::<_, EmbeddingWorkerError>(());
                    tracing::debug!(
                        "update gradient embedding worker time cost {:?}",
                        start_time.elapsed()
                    );
                    result
                }
            });

        let _updated_gradient_groups: Vec<_> = futures::future::try_join_all(futs).await?;

        tracing::debug!(
            "update gradients all slots time cost {:?}",
            start_time.elapsed()
        );

        if let Ok(m) = MetricsHolder::get() {
            m.update_gradient_time_cost_sec
                .set(start_time.elapsed().as_secs_f64());
        }

        Ok(())
    }

    pub async fn lookup_batched_all_slots(
        &self,
        indices: &mut IDTypeFeatureBatch,
        requires_grad: bool,
    ) -> Result<Vec<FeatureEmbeddingBatch>, EmbeddingWorkerError> {
        let start_time_all = std::time::Instant::now();
        let start_time = std::time::Instant::now();

        let all_shards_ids = tokio::task::block_in_place(|| {
            lookup_batched_all_slots_preprocess(indices, &self.embedding_config, self.replica_size)
        });

        let futs = all_shards_ids
            .into_iter()
            .enumerate()
            .map(|(replica_index, shard_indices)| {
                let req = tokio::task::block_in_place(|| {
                    (
                        shard_indices.iter().map(|x| (x.sign, x.dim)).collect(),
                        requires_grad,
                    )
                });
                let client = block_on(
                    self.all_embedding_server_client
                        .get_client_by_index(replica_index),
                );
                async move {
                    let lookup_results: Vec<f32> = client
                        .lookup_mixed(&req)
                        .await
                        .map_err(|e| EmbeddingWorkerError::RpcError(format!("{:?}", e)))??;
                    Ok::<_, EmbeddingWorkerError>((lookup_results, shard_indices))
                }
            });

        tracing::debug!(
            "create sharded requests time cost {:?}",
            start_time.elapsed()
        );
        if let Ok(m) = MetricsHolder::get() {
            m.lookup_create_requests_time_cost_sec
                .set(start_time.elapsed().as_secs_f64());
        }
        let start_time = std::time::Instant::now();

        let forwarded_groups: Vec<_> = futures::future::try_join_all(futs).await?;

        tracing::debug!("rpc time cost {:?}", start_time.elapsed());
        if let Ok(m) = MetricsHolder::get() {
            m.lookup_rpc_time_cost_sec
                .set(start_time.elapsed().as_secs_f64());
        }

        let start_time = std::time::Instant::now();

        let batches = tokio::task::block_in_place(|| {
            lookup_batched_all_slots_postprocess(indices, forwarded_groups, &self.embedding_config)
        });

        tracing::debug!("summation time cost {:?}", start_time.elapsed());
        if let Ok(m) = MetricsHolder::get() {
            m.summation_time_cost_sec
                .set(start_time.elapsed().as_secs_f64());
            m.lookup_batched_time_cost_sec
                .set(start_time_all.elapsed().as_secs_f64());
        }

        return Ok(batches);
    }

    pub async fn ready_for_serving(&self) -> bool {
        let result = self.all_embedding_server_client.ready_for_serving().await;
        tracing::info!("embedding worker server ready for serving: {}", result);
        result
    }

    pub async fn model_manager_status(&self) -> Vec<EmbeddingModelManagerStatus> {
        let result = self
            .all_embedding_server_client
            .model_manager_status()
            .await;
        tracing::info!("embedding server dumping model: {:?}", result);
        result
    }

    pub async fn set_embedding(
        &self,
        req: Vec<HashMapEmbeddingEntry>,
    ) -> Result<(), EmbeddingWorkerError> {
        let replica_size = self.replica_size;
        let futs: Vec<_> = tokio::task::block_in_place(|| {
            let grouped_entries = req
                .into_iter()
                .sorted_by_key(|e| sign_to_shard_modulo(e.sign(), replica_size))
                .group_by(|e| sign_to_shard_modulo(e.sign(), replica_size));

            grouped_entries
                .into_iter()
                .map(|(replica_index, requests)| {
                    let group = requests.into_iter().collect_vec();
                    let client = block_on(
                        self.all_embedding_server_client
                            .get_client_by_index(replica_index as usize),
                    );
                    async move {
                        client
                            .set_embedding(&group)
                            .await
                            .map_err(|e| EmbeddingWorkerError::RpcError(format!("{:?}", e)))??;
                        Ok::<_, EmbeddingWorkerError>(())
                    }
                })
                .collect()
        });
        futures::future::try_join_all(futs).await.map(|_| ())
    }

    pub async fn can_forward_batched(&self, batcher_idx: usize) -> bool {
        let result = match self.forward_id_buffer.read().await.get(&batcher_idx) {
            Some(buffer) => buffer.len() < self.embedding_worker_config.forward_buffer_size,
            None => true,
        };
        let t = self.cannot_forward_batched_time.load();
        if !result {
            let current_time = SystemTime::now();
            if current_time.duration_since(t).unwrap() > Duration::from_secs(60) {
                let mut forward_id_buffer = self.forward_id_buffer.write().await;
                let sub_buffer = forward_id_buffer.get_mut(&batcher_idx).unwrap();
                tokio::task::block_in_place(|| {
                    let old_keys = sub_buffer
                        .iter()
                        .filter_map(|(k, v)| {
                            if current_time
                                .duration_since(v.enter_forward_id_buffer_time.unwrap())
                                .unwrap()
                                > Duration::from_secs(
                                    self.embedding_worker_config.buffered_data_expired_sec as u64,
                                )
                            {
                                Some(*k)
                            } else {
                                None
                            }
                        })
                        .collect_vec();
                    for k in old_keys {
                        sub_buffer.remove(&k);
                    }
                    self.cannot_forward_batched_time.store(SystemTime::now());
                });
            }
        } else {
            self.cannot_forward_batched_time.store(SystemTime::now());
        }
        result
    }

    pub async fn forward_batch_id(
        &self,
        req: (IDTypeFeatureRemoteRef, bool),
    ) -> Result<EmbeddingBatch, EmbeddingWorkerError> {
        let (id_type_feature_remote_ref, requires_grad) = req;
        let ref_id = id_type_feature_remote_ref.ref_id;

        let inner = self.clone();
        let mut indices = {
            let mut forward_id_buffer = inner.forward_id_buffer.write().await;
            let sub_buffer = forward_id_buffer
                .get_mut(&id_type_feature_remote_ref.batcher_idx)
                .ok_or_else(|| EmbeddingWorkerError::ForwardIdNotFound(ref_id))?;
            sub_buffer
                .remove(&ref_id)
                .ok_or_else(|| EmbeddingWorkerError::ForwardIdNotFound(ref_id))?
        };

        tracing::debug!("received forward_batch_id request");
        self.staleness.fetch_add(1, Ordering::AcqRel);
        let result = inner
            .lookup_batched_all_slots(&mut indices, requires_grad)
            .await;

        if result.is_err() {
            self.staleness.fetch_sub(1, Ordering::AcqRel);
        }
        let result = result?;

        if requires_grad {
            indices.enter_post_forward_buffer_time = Some(SystemTime::now());
            inner
                .post_forward_buffer
                .write()
                .await
                .insert(ref_id, indices);
            tracing::debug!("indices inserted into post forward buffer");
        }

        return Ok(EmbeddingBatch {
            batches: result,
            backward_ref_id: Some(ref_id),
        });
    }

    pub async fn forward_batched_direct(
        &self,
        indices: IDTypeFeatureBatch,
    ) -> Result<EmbeddingBatch, EmbeddingWorkerError> {
        let mut indices = indices;

        let requires_grad = indices.requires_grad.clone();
        let result = self
            .lookup_batched_all_slots(&mut indices, requires_grad)
            .await?;

        let backward_ref_id = if requires_grad {
            let backward_ref_id = self.get_id();
            let inner = self.clone();

            indices.enter_post_forward_buffer_time = Some(SystemTime::now());
            inner
                .post_forward_buffer
                .write()
                .await
                .insert(backward_ref_id, indices);
            tracing::debug!("indices inserted into post forward buffer");
            Some(backward_ref_id)
        } else {
            None
        };

        Ok(EmbeddingBatch {
            batches: result,
            backward_ref_id,
        })
    }

    pub async fn update_gradient_batched(
        &self,
        req: (u64, EmbeddingGradientBatch),
    ) -> Result<(), EmbeddingWorkerError> {
        let (backward_ref_id, mut gradients) = req;
        let indices = self
            .post_forward_buffer
            .write()
            .await
            .remove(&backward_ref_id)
            .ok_or_else(|| EmbeddingWorkerError::ForwardIdNotFound(backward_ref_id))?;

        let inner = self.clone();
        inner
            .update_all_batched_gradients(&mut gradients, indices)
            .await?;

        self.staleness.fetch_sub(1, Ordering::AcqRel);

        Ok(())
    }

    pub async fn dump(&self, req: String) -> Result<(), EmbeddingWorkerError> {
        let inner = self.clone();
        let futs = (0..inner.all_embedding_server_client.replica_size()).map(|client_idx| {
            let req = req.clone();
            async move {
                let client = inner
                    .all_embedding_server_client
                    .get_client_by_index(client_idx)
                    .await;
                client
                    .dump(&req)
                    .await
                    .map_err(|e| EmbeddingWorkerError::RpcError(e.to_string()))??;
                Ok(())
            }
        });
        futures::future::try_join_all(futs).await.map(|_| ())
    }

    pub async fn load(&self, req: String) -> Result<(), EmbeddingWorkerError> {
        let emb_dir = PathBuf::from(req.clone());
        let model_info = self
            .embedding_model_manager
            .load_embedding_checkpoint_info(&emb_dir)?;
        if model_info.num_shards == self.all_embedding_server_client.dst_replica_size {
            tracing::info!("loading embedding from {} via embedding servers", req);
            self.load_embedding_via_emb_servers(req).await?;
        } else {
            tracing::info!("loading embedding from {} via embedding worker", req);
            self.load_embedding_via_embedding_worker(req, model_info.num_shards)
                .await?;
        }
        Ok(())
    }

    pub async fn load_embedding_via_emb_servers(
        &self,
        req: String,
    ) -> Result<(), EmbeddingWorkerError> {
        if !self.is_master_server()? {
            return Ok(());
        }
        let inner = self.clone();
        let futs = (0..inner.all_embedding_server_client.replica_size()).map(|client_idx| {
            let req = req.clone();
            async move {
                let client = inner
                    .all_embedding_server_client
                    .get_client_by_index(client_idx)
                    .await;
                client
                    .load(&req)
                    .await
                    .map_err(|e| EmbeddingWorkerError::RpcError(e.to_string()))??;
                Ok(())
            }
        });
        let result = futures::future::try_join_all(futs).await.map(|_| ());
        result
    }

    pub async fn load_embedding_via_embedding_worker(
        &self,
        req: String,
        num_model_shards: usize,
    ) -> Result<(), EmbeddingWorkerError> {
        let root_dir = PathBuf::from(req);
        let repilca_info = PersiaReplicaInfo::get()?;
        let mut dst_shard_idx = repilca_info.replica_index;

        let mut emb_file_list: Vec<PathBuf> = Vec::new();
        while dst_shard_idx < num_model_shards {
            let shard_dir = self
                .embedding_model_manager
                .get_other_shard_dir(&root_dir, dst_shard_idx);
            let mut shard_file_list = self
                .embedding_model_manager
                .get_emb_file_list_in_dir(shard_dir)?;
            emb_file_list.append(&mut shard_file_list);

            dst_shard_idx += repilca_info.replica_size;
        }

        tracing::debug!("embedding filelist: {:?}", emb_file_list);

        if emb_file_list.len() == 0 {
            return Ok(());
        }

        let num_checkpointing_workers = PersiaCommonConfig::get()?.checkpointing_config.num_workers;
        let num_file_per_worker = emb_file_list.len() / num_checkpointing_workers;

        let num_files = emb_file_list.len();
        let loaded = Arc::new(AtomicUsize::new(0));

        let grouped_emb_file_list: Vec<Vec<PathBuf>> = emb_file_list
            .into_iter()
            .chunks(num_file_per_worker)
            .into_iter()
            .map(|chunk| chunk.collect())
            .collect();

        let futs: Vec<_> = grouped_emb_file_list
            .into_iter()
            .map(|file_list| {
                let embedding_worker_inner = self.clone();
                let embedding_model_manager = self.embedding_model_manager.clone();
                let loaded = loaded.clone();
                async move {
                    for file_path in file_list.into_iter() {
                        let array_linked_list = tokio::task::block_in_place(|| {
                            embedding_model_manager.load_array_linked_list(file_path)
                        })?;
                        let entries = Vec::from_iter(array_linked_list.into_iter());
                        embedding_worker_inner.set_embedding(entries).await?;
                        let cur_loaded = loaded.fetch_add(1, Ordering::AcqRel) + 1;
                        let progress = (cur_loaded as f32 / num_files as f32) * 100.0_f32;
                        tracing::info!(
                            "loading embedding via embedding worker, pregress: {}%",
                            progress
                        );
                    }
                    Ok(())
                }
            })
            .collect();

        futures::future::try_join_all(futs).await.map(|_| ())
    }

    pub async fn configure_embedding_parameter_servers(
        &self,
        req: PersiaEmbeddingModelHyperparameters,
    ) -> Result<(), EmbeddingWorkerError> {
        let inner = self.clone();
        let req = req;
        let futs = (0..inner.all_embedding_server_client.replica_size()).map(|client_idx| {
            let req = req.clone();
            async move {
                let client = inner
                    .all_embedding_server_client
                    .get_client_by_index(client_idx)
                    .await;
                client
                    .configure(&req)
                    .await
                    .map_err(|e| EmbeddingWorkerError::RpcError(e.to_string()))??;
                Ok(())
            }
        });
        let result = futures::future::try_join_all(futs).await.map(|_| ());
        tracing::info!("embedding servers configured: {:?}", result);
        result
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), EmbeddingWorkerError> {
        let inner = self.clone();
        let futs = (0..inner.all_embedding_server_client.replica_size()).map(|client_idx| {
            let optimizer = optimizer.clone();
            async move {
                let client = inner
                    .all_embedding_server_client
                    .get_client_by_index(client_idx)
                    .await;
                client
                    .register_optimizer(&optimizer)
                    .await
                    .map_err(|e| EmbeddingWorkerError::RpcError(e.to_string()))??;
                Ok(())
            }
        });
        tracing::info!("register optimizer: {:?}", &optimizer);
        futures::future::try_join_all(futs).await.map(|_| ())
    }

    pub async fn get_address(&self) -> Result<String, EmbeddingWorkerError> {
        let instance_info = InstanceInfo::get()?;
        let address = format!("{}:{}", instance_info.ip_address, instance_info.port);
        Ok(address)
    }

    pub async fn get_replica_size(&self) -> Result<usize, EmbeddingWorkerError> {
        let repilca_info = PersiaReplicaInfo::get()?;
        Ok(repilca_info.replica_size)
    }

    pub async fn error_handle(
        &self,
        err: &EmbeddingWorkerError,
    ) -> Result<(), EmbeddingWorkerError> {
        match err {
            EmbeddingWorkerError::RpcError(_) => {
                let servers = self.all_embedding_server_client.get_all_addresses().await?;
                self.all_embedding_server_client
                    .update_rpc_clients(servers, false)
                    .await
            }
            _ => Ok(()),
        }
    }

    pub async fn get_embedding_size(&self) -> Result<Vec<usize>, EmbeddingWorkerError> {
        let inner = self.clone();
        let futs =
            (0..inner.all_embedding_server_client.replica_size()).map(|client_idx| async move {
                let client = inner
                    .all_embedding_server_client
                    .get_client_by_index(client_idx)
                    .await;
                let result = client
                    .get_embedding_size(&())
                    .await
                    .map_err(|e| EmbeddingWorkerError::RpcError(format!("{:?}", e)))??;
                Ok(result)
            });

        let result = futures::future::try_join_all(futs).await;
        result
    }

    pub async fn clear_embeddings(&self) -> Result<(), EmbeddingWorkerError> {
        let inner = self.clone();
        let futs =
            (0..inner.all_embedding_server_client.replica_size()).map(|client_idx| async move {
                let client = inner
                    .all_embedding_server_client
                    .get_client_by_index(client_idx)
                    .await;
                client
                    .clear_embeddings(&())
                    .await
                    .map_err(|e| EmbeddingWorkerError::RpcError(format!("{:?}", e)))??;
                Ok(())
            });
        futures::future::try_join_all(futs).await.map(|_| ())
    }
}

#[derive(Clone)]
pub struct EmbeddingWorker {
    pub inner: Arc<EmbeddingWorkerInner>,
    pub shutdown_channel:
        Arc<persia_libs::async_lock::RwLock<Option<tokio::sync::oneshot::Sender<()>>>>,
}

#[persia_rpc_macro::service]
impl EmbeddingWorker {
    pub async fn ready_for_serving(&self, _req: ()) -> bool {
        self.inner.ready_for_serving().await
    }

    pub async fn model_manager_status(&self, _req: ()) -> Vec<EmbeddingModelManagerStatus> {
        self.inner.model_manager_status().await
    }

    pub async fn set_embedding(
        &self,
        req: Vec<HashMapEmbeddingEntry>,
    ) -> Result<(), EmbeddingWorkerError> {
        self.inner.set_embedding(req).await
    }

    pub async fn get_embedding_size(&self, _req: ()) -> Result<Vec<usize>, EmbeddingWorkerError> {
        self.inner.get_embedding_size().await
    }

    pub async fn clear_embeddings(&self, _req: ()) -> Result<(), EmbeddingWorkerError> {
        self.inner.clear_embeddings().await
    }

    pub async fn shutdown_server(&self, _req: ()) -> Result<(), EmbeddingParameterServerError> {
        let futs = (0..self.inner.all_embedding_server_client.replica_size()).map(
            |client_idx| async move {
                let client = self
                    .inner
                    .all_embedding_server_client
                    .get_client_by_index(client_idx)
                    .await;
                client.shutdown(&()).await
            },
        );

        let result = futures::future::try_join_all(futs).await;

        if result.is_ok() {
            Ok(())
        } else {
            Err(EmbeddingParameterServerError::ShutdownError)
        }
    }

    pub async fn shutdown(&self, _req: ()) -> Result<(), EmbeddingWorkerError> {
        let mut shutdown_channel = self.shutdown_channel.write().await;
        let shutdown_channel = shutdown_channel.take();

        match shutdown_channel {
            Some(sender) => {
                sender.send(()).unwrap();
                Ok(())
            }
            None => {
                tracing::debug!("shutdown channel already been taken, wait server shutdown.");
                Ok(())
            }
        }
    }

    pub async fn forward_batch_id(
        &self,
        req: (IDTypeFeatureRemoteRef, bool),
    ) -> Result<EmbeddingBatch, EmbeddingWorkerError> {
        let resp = self.inner.forward_batch_id(req).await;
        if resp.is_err() {
            self.inner.error_handle(resp.as_ref().unwrap_err()).await?;
        }
        resp
    }

    pub async fn forward_batched_direct(
        &self,
        indices: IDTypeFeatureBatch,
    ) -> Result<EmbeddingBatch, EmbeddingWorkerError> {
        self.inner.forward_batched_direct(indices).await
    }

    pub async fn update_gradient_batched(
        &self,
        req: (u64, EmbeddingGradientBatch),
    ) -> Result<(), EmbeddingWorkerError> {
        let resp = self.inner.update_gradient_batched(req).await;
        if resp.is_err() {
            self.inner.error_handle(resp.as_ref().unwrap_err()).await?;
        }
        resp
    }

    pub async fn dump(&self, req: String) -> Result<(), EmbeddingWorkerError> {
        self.inner.dump(req).await
    }

    pub async fn load(&self, req: String) -> Result<(), EmbeddingWorkerError> {
        self.inner.load(req).await
    }

    pub async fn configure_embedding_parameter_servers(
        &self,
        req: PersiaEmbeddingModelHyperparameters,
    ) -> Result<(), EmbeddingWorkerError> {
        self.inner.configure_embedding_parameter_servers(req).await
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), EmbeddingWorkerError> {
        self.inner.register_optimizer(optimizer).await
    }
}

#[derive(Clone)]
pub struct EmbeddingWorkerNatsService {
    pub inner: Arc<EmbeddingWorkerInner>,
}

#[persia_nats_marcos::service]
impl EmbeddingWorkerNatsService {
    pub async fn ready_for_serving(&self, _req: ()) -> bool {
        self.inner.ready_for_serving().await
    }

    pub async fn model_manager_status(&self, _req: ()) -> Vec<EmbeddingModelManagerStatus> {
        self.inner.model_manager_status().await
    }

    pub async fn can_forward_batched(&self, batcher_idx: usize) -> bool {
        self.inner.can_forward_batched(batcher_idx).await
    }

    pub async fn forward_batched(
        &self,
        indices: IDTypeFeatureBatch,
    ) -> Result<IDTypeFeatureRemoteRef, EmbeddingWorkerError> {
        let batcher_idx = indices
            .batcher_idx
            .ok_or_else(|| EmbeddingWorkerError::DataSrcIdxNotSet)?;
        if !self.inner.can_forward_batched(batcher_idx).await {
            return Err(EmbeddingWorkerError::ForwardBufferFull);
        }
        let ref_id = self.inner.forward_batched(indices, batcher_idx).await?;
        let embedding_worker_addr = self.inner.get_address().await?;
        let id_type_feature_remote_ref = IDTypeFeatureRemoteRef {
            embedding_worker_addr,
            ref_id,
            batcher_idx,
        };
        Ok(id_type_feature_remote_ref)
    }

    pub async fn dump(&self, req: String) -> Result<(), EmbeddingWorkerError> {
        self.inner.dump(req).await
    }

    pub async fn load(&self, req: String) -> Result<(), EmbeddingWorkerError> {
        self.inner.load(req).await
    }

    pub async fn configure_embedding_parameter_servers(
        &self,
        req: PersiaEmbeddingModelHyperparameters,
    ) -> Result<(), EmbeddingWorkerError> {
        self.inner.configure_embedding_parameter_servers(req).await
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), EmbeddingWorkerError> {
        self.inner.register_optimizer(optimizer).await
    }

    pub async fn get_address(&self, _req: ()) -> Result<String, EmbeddingWorkerError> {
        self.inner.get_address().await
    }

    pub async fn get_replica_size(&self, _req: ()) -> Result<usize, EmbeddingWorkerError> {
        self.inner.get_replica_size().await
    }
}

#[cfg(test)]
mod lookup_batched_all_slots_preprocess_tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use persia_common::FeatureBatch;
    use persia_libs::serde_yaml;

    #[test]
    fn test_indices_to_hashstack_indices() {
        let config = "feature_index_prefix_bit: 12\nslots_config:\n  Test:\n    dim: 32\n    hash_stack_config:\n      hash_stack_rounds: 2\n      embedding_size: 10\nfeature_groups: {}\n";

        let config: EmbeddingConfig = serde_yaml::from_str(config).expect("failed to parse config");

        let raw_batch: Vec<Vec<u64>> = vec![vec![12, 23, 34], vec![56, 78, 90], vec![12, 56]];
        let feature_name = "Test".to_string();
        let feature_batch = FeatureBatch::new(feature_name.clone(), raw_batch);
        let mut id_type_feature_batch = IDTypeFeatureBatch {
            requires_grad: false,
            batches: vec![feature_batch],
            enter_forward_id_buffer_time: None,
            enter_post_forward_buffer_time: None,
            batcher_idx: None,
        };
        indices_to_hashstack_indices(&mut id_type_feature_batch, &config);
        let hashstack_feature_batch = id_type_feature_batch.batches.first().unwrap();

        let target_raw_batch: Vec<Vec<u64>> = vec![
            vec![2, 18, 5, 10, 0, 11],
            vec![6, 17, 7, 12, 8, 16],
            vec![2, 18, 6, 17],
        ];
        let target_feature_batch = FeatureBatch::new(feature_name, target_raw_batch);

        for single_sign in hashstack_feature_batch.index_batch.iter() {
            for target_single_sign in target_feature_batch.index_batch.iter() {
                if single_sign.sign == target_single_sign.sign {
                    let mut result = single_sign.in_which_batch_samples.clone();
                    let mut target = target_single_sign.in_which_batch_samples.clone();
                    assert_eq!(result.len(), target.len());
                    result.sort();
                    target.sort();
                    let matching = result
                        .iter()
                        .zip(&target)
                        .filter(|&((r, _), (t, _))| r == t)
                        .count();
                    assert_eq!(matching, target.len());
                }
            }
        }
    }

    #[test]
    fn test_indices_add_prefix() {
        let config = "feature_index_prefix_bit: 12\nslots_config:\n  feature1:\n    dim: 64\n    index_prefix: 450359962737049600\n";

        let config: EmbeddingConfig = serde_yaml::from_str(config).expect("failed to parse config");

        let mut raw_batch: Vec<Vec<u64>> = vec![
            vec![12, 23, 34],
            vec![56, 78, 90],
            vec![16000000000000000, 56],
        ];
        let feature_name = "feature1".to_string();
        let feature_batch = FeatureBatch::new(feature_name.clone(), raw_batch.clone());
        let mut id_type_feature_batch = IDTypeFeatureBatch {
            requires_grad: false,
            batches: vec![feature_batch],
            enter_forward_id_buffer_time: None,
            enter_post_forward_buffer_time: None,
            batcher_idx: None,
        };
        indices_add_prefix(&mut id_type_feature_batch, &config);
        let result_feature_batch = id_type_feature_batch.batches.first().unwrap();

        result_feature_batch.index_batch.iter().for_each(|x| {
            x.in_which_batch_samples
                .iter()
                .for_each(|(batch_idx, col_idx)| {
                    raw_batch[*batch_idx as usize][*col_idx as usize] = x.sign;
                })
        });

        let target_raw_batch: Vec<Vec<u64>> = vec![
            vec![450359962737049612, 450359962737049623, 450359962737049634],
            vec![450359962737049656, 450359962737049678, 450359962737049690],
            vec![452849163854938115, 450359962737049656],
        ];

        raw_batch
            .iter()
            .zip(target_raw_batch.iter())
            .for_each(|(result, target)| {
                result.iter().zip(target.iter()).for_each(|(r, t)| {
                    assert_eq!(r, t);
                })
            });
    }
}
