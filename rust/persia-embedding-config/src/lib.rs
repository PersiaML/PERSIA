#![allow(clippy::needless_return)]
#[allow(dead_code)]
pub mod feature_config;
use once_cell::sync::OnceCell;
use persia_speedy::{Readable, Writable};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use thiserror::Error;

#[derive(Readable, Writable, Error, Debug, Clone)]
pub enum PersiaGlobalConfigError {
    #[error("global config not ready error")]
    NotReadyError,
    #[error("global config set error")]
    SetError,
    #[error("failed to open/read config file error")]
    ConfigFileError(String),
}

#[derive(Serialize, Deserialize, Readable, Writable, Debug, Default, Clone)]
pub struct BoundedUniformInitialization {
    pub lower: f32,
    pub upper: f32,
}

impl BoundedUniformInitialization {
    pub fn new(lower: f32, upper: f32) -> Self {
        BoundedUniformInitialization { lower, upper }
    }
}

#[derive(Serialize, Deserialize, Readable, Writable, Debug, Default, Clone)]
pub struct BoundedGammaInitialization {
    pub shape: f32,
    pub scale: f32,
}

impl BoundedGammaInitialization {
    pub fn new(shape: f32, scale: f32) -> Self {
        BoundedGammaInitialization { shape, scale }
    }
}

#[derive(Serialize, Deserialize, Readable, Writable, Debug, Default, Clone)]
pub struct BoundedPoissonInitialization {
    pub lambda: f32,
}

impl BoundedPoissonInitialization {
    pub fn new(lambda: f32) -> Self {
        BoundedPoissonInitialization { lambda }
    }
}

#[derive(Serialize, Deserialize, Readable, Writable, Debug, Default, Clone)]
pub struct BoundedNormalInitialization {
    pub mean: f32,
    pub standard_deviation: f32,
}

impl BoundedNormalInitialization {
    pub fn new(mean: f32, standard_deviation: f32) -> Self {
        BoundedNormalInitialization {
            mean,
            standard_deviation,
        }
    }
}

#[derive(Serialize, Deserialize, Readable, Writable, Debug, Clone)]
pub enum InitializationMethod {
    InverseEmbeddingSizeSqrt,
    BoundedUniform(BoundedUniformInitialization),
    BoundedGamma(BoundedGammaInitialization),
    BoundedPoisson(BoundedPoissonInitialization),
    BoundedNormal(BoundedNormalInitialization),
}

impl Default for InitializationMethod {
    fn default() -> Self {
        Self::BoundedUniform(BoundedUniformInitialization {
            lower: -0.01,
            upper: 0.01,
        })
    }
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct PersiaSparseModelHyperparameters {
    pub initialization_method: InitializationMethod,
    pub admit_probability: f32,
    pub weight_bound: f32,
    pub enable_weight_bound: bool,
}

static PERSIA_EMBEDDING_SEVER_CONFIG: OnceCell<Arc<PersiaShardedServerConfig>> = OnceCell::new();

static PERSIA_MIDDLEWARE_CONFIG: OnceCell<Arc<PersiaMiddlewareConfig>> = OnceCell::new();

static PERSIA_COMMON_CONFIG: OnceCell<Arc<PersiaCommonConfig>> = OnceCell::new();

static PERSIA_REPLICA_INFO: OnceCell<Arc<PersiaReplicaInfo>> = OnceCell::new();

static PERSIA_INSTANCE_INFO: OnceCell<Arc<InstanceInfo>> = OnceCell::new();

static PERSIA_EMBEDDING_CONFIG: OnceCell<Arc<EmbeddingConfig>> = OnceCell::new();

fn get_local_ip() -> String {
    let if_addrs = get_if_addrs::get_if_addrs().expect("failed to get local ip address");

    let persia_socket_name = std::env::var("PERSIA_SOCKET_NAME").unwrap_or(String::from("eth0"));

    if let Some(ip_addr) = if_addrs
        .iter()
        .find(|x| x.name == persia_socket_name.as_str())
    {
        let ip_addr = ip_addr.ip().to_string();
        return ip_addr;
    } else {
        panic!(
            "failed to get ip of socket {}, please try other by setting PERSIA_SOCKET_NAME",
            persia_socket_name
        );
    }
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct InferConfig {
    pub servers: Vec<String>,
    pub embedding_checkpoint: String,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub enum PerisaIntent {
    Train,
    Eval,
    Infer(InferConfig),
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub enum PersiaPersistenceStorage {
    Ceph,
    Hdfs,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct PersiaReplicaInfo {
    pub replica_size: usize,
    pub replica_index: usize,
}

impl PersiaReplicaInfo {
    pub fn get() -> Result<Arc<Self>, PersiaGlobalConfigError> {
        let singleton = PERSIA_REPLICA_INFO.get();
        match singleton {
            Some(s) => Ok(s.clone()),
            None => Err(PersiaGlobalConfigError::NotReadyError),
        }
    }
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct InstanceInfo {
    pub ip_address: String,
    pub port: u16,
}

impl InstanceInfo {
    pub fn new(port: u16) -> Self {
        Self {
            ip_address: get_local_ip(),
            port,
        }
    }

    pub fn get() -> Result<Arc<Self>, PersiaGlobalConfigError> {
        let singleton = PERSIA_INSTANCE_INFO.get();
        match singleton {
            Some(s) => Ok(s.clone()),
            None => Err(PersiaGlobalConfigError::NotReadyError),
        }
    }
}

fn get_true() -> bool {
    true
}

fn get_false() -> bool {
    false
}

fn get_four() -> usize {
    4
}

fn get_eight() -> usize {
    8
}

fn get_zero() -> u64 {
    0
}

fn get_ten() -> usize {
    10
}

fn get_hundred() -> usize {
    100
}

fn get_thousand() -> usize {
    1000
}

fn get_million() -> usize {
    1_000_000
}

fn get_billion() -> usize {
    1_000_000_000
}

fn get_default_local_buffer_dir() -> String {
    String::from("/workspace/buffer_dir/")
}

fn get_default_incremental_dir() -> String {
    String::from("/workspace/incremental_dir/")
}

fn get_default_job_name() -> String {
    String::from("persia_default_jobname")
}

fn get_default_storage() -> PersiaPersistenceStorage {
    PersiaPersistenceStorage::Ceph
}

fn get_default_common_config() -> PersiaCommonConfig {
    PersiaCommonConfig::default()
}

fn get_default_middleware_config() -> PersiaMiddlewareConfig {
    PersiaMiddlewareConfig::default()
}

fn get_default_shard_server_config() -> PersiaShardedServerConfig {
    PersiaShardedServerConfig::default()
}

fn get_default_metrics_config() -> PersiaMetricsConfig {
    PersiaMetricsConfig::default()
}

fn get_default_intent() -> PerisaIntent {
    PerisaIntent::Train
}

fn get_default_hashstack_config() -> HashStackConfig {
    HashStackConfig {
        hash_stack_rounds: 0,
        embedding_size: 0,
    }
}

fn get_default_feature_groups() -> indexmap::IndexMap<String, Vec<String>> {
    indexmap::IndexMap::new()
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct PersiaMetricsConfig {
    #[serde(default = "get_false")]
    pub enable_metrics: bool,
    #[serde(default = "get_default_job_name")]
    pub job_name: String,
    #[serde(default = "get_ten")]
    pub push_interval_seconds: usize,
}

impl Default for PersiaMetricsConfig {
    fn default() -> Self {
        Self {
            enable_metrics: false,
            job_name: get_default_job_name(),
            push_interval_seconds: 10,
        }
    }
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct PersiaCommonConfig {
    #[serde(default = "get_default_metrics_config")]
    pub metrics_config: PersiaMetricsConfig,
    #[serde(default = "get_default_intent")]
    pub intent: PerisaIntent,
}

impl Default for PersiaCommonConfig {
    fn default() -> Self {
        Self {
            metrics_config: PersiaMetricsConfig::default(),
            intent: PerisaIntent::Train,
        }
    }
}

impl PersiaCommonConfig {
    pub fn get() -> Result<Arc<Self>, PersiaGlobalConfigError> {
        let singleton = PERSIA_COMMON_CONFIG.get();
        match singleton {
            Some(s) => Ok(s.clone()),
            None => Err(PersiaGlobalConfigError::NotReadyError),
        }
    }
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct PersiaMiddlewareConfig {
    #[serde(default = "get_thousand")]
    pub forward_buffer_size: usize,
    #[serde(default = "get_thousand")]
    pub buffered_data_expired_sec: usize,
}

impl Default for PersiaMiddlewareConfig {
    fn default() -> Self {
        Self {
            forward_buffer_size: 1000,
            buffered_data_expired_sec: 1000,
        }
    }
}

impl PersiaMiddlewareConfig {
    pub fn get() -> Result<Arc<Self>, PersiaGlobalConfigError> {
        let singleton = PERSIA_MIDDLEWARE_CONFIG.get();
        match singleton {
            Some(s) => Ok(s.clone()),
            None => Err(PersiaGlobalConfigError::NotReadyError),
        }
    }
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct PersiaShardedServerConfig {
    // Eviction map config
    #[serde(default = "get_billion")]
    pub capacity: usize,
    #[serde(default = "get_hundred")]
    pub num_hashmap_internal_shards: usize,
    #[serde(default = "get_thousand")]
    pub full_amount_manager_buffer_size: usize,
    #[serde(default = "get_million")]
    pub embedding_recycle_pool_capacity: usize,

    // model persistence config
    #[serde(default = "get_four")]
    pub num_persistence_workers: usize,
    #[serde(default = "get_million")]
    pub num_signs_per_file: usize,

    #[serde(default = "get_default_storage")]
    pub storage: PersiaPersistenceStorage,
    #[serde(default = "get_default_local_buffer_dir")]
    pub local_buffer_dir: String,

    // incremental dump config
    #[serde(default = "get_false")]
    pub enable_incremental_update: bool,
    #[serde(default = "get_million")]
    pub incremental_buffer_size: usize,
    #[serde(default = "get_default_incremental_dir")]
    pub incremental_dir: String,
    #[serde(default = "get_thousand")]
    pub incremental_channel_capacity: usize,
}

impl Default for PersiaShardedServerConfig {
    fn default() -> Self {
        Self {
            capacity: 1_000_000_000,
            num_hashmap_internal_shards: 1000,
            full_amount_manager_buffer_size: 1000,
            embedding_recycle_pool_capacity: 1_000_000,
            num_persistence_workers: 4,
            num_signs_per_file: 1_000_000,
            storage: get_default_storage(),
            local_buffer_dir: get_default_local_buffer_dir(),
            enable_incremental_update: false,
            incremental_buffer_size: 1_000_000,
            incremental_dir: get_default_incremental_dir(),
            incremental_channel_capacity: 1000,
        }
    }
}

impl PersiaShardedServerConfig {
    pub fn get() -> Result<Arc<Self>, PersiaGlobalConfigError> {
        let singleton = PERSIA_EMBEDDING_SEVER_CONFIG.get();
        match singleton {
            Some(s) => Ok(s.clone()),
            None => Err(PersiaGlobalConfigError::NotReadyError),
        }
    }
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct PersiaGlobalConfig {
    #[serde(default = "get_default_common_config")]
    pub common_config: PersiaCommonConfig,
    #[serde(default = "get_default_middleware_config")]
    pub middleware_config: PersiaMiddlewareConfig,
    #[serde(default = "get_default_shard_server_config")]
    pub shard_server_config: PersiaShardedServerConfig,
}

impl PersiaGlobalConfig {
    pub fn set_configures(
        file_path: &PathBuf,
        port: u16,
        replica_index: usize,
        replica_size: usize,
    ) -> Result<(), PersiaGlobalConfigError> {
        if !file_path.is_file() {
            tracing::error!("global config yaml file NOT found");
            std::thread::sleep(std::time::Duration::from_secs(120));
            panic!("global config yaml file NOT found")
        }

        let global_config: PersiaGlobalConfig = serde_yaml::from_reader(
            std::fs::File::open(file_path).expect("cannot read config file"),
        )
        .expect("cannot parse config file");

        tracing::info!(
            "setting shard_server_config {:?}",
            global_config.shard_server_config
        );
        PERSIA_EMBEDDING_SEVER_CONFIG
            .set(Arc::new(global_config.shard_server_config))
            .map_err(|_| PersiaGlobalConfigError::SetError)?;

        tracing::info!(
            "setting middleware_config {:?}",
            global_config.middleware_config
        );
        PERSIA_MIDDLEWARE_CONFIG
            .set(Arc::new(global_config.middleware_config))
            .map_err(|_| PersiaGlobalConfigError::SetError)?;

        tracing::info!("setting common_config {:?}", global_config.common_config);
        PERSIA_COMMON_CONFIG
            .set(Arc::new(global_config.common_config))
            .map_err(|_| PersiaGlobalConfigError::SetError)?;

        let instance_info = InstanceInfo::new(port);
        tracing::info!("setting instance_info {:?}", instance_info);
        PERSIA_INSTANCE_INFO
            .set(Arc::new(instance_info))
            .map_err(|_| PersiaGlobalConfigError::SetError)?;

        let replica_info = PersiaReplicaInfo {
            replica_index,
            replica_size,
        };
        tracing::info!("setting replica_info {:?}", replica_info);
        PERSIA_REPLICA_INFO
            .set(Arc::new(replica_info))
            .map_err(|_| PersiaGlobalConfigError::SetError)?;

        Ok(())
    }
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct HashStackConfig {
    pub hash_stack_rounds: usize,
    pub embedding_size: usize,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct SlotConfig {
    pub dim: usize,
    #[serde(default = "get_ten")]
    pub sample_fixed_size: usize, // raw embedding placeholder size to fill 3d tensor -> (bs, sample_fix_sized, dim)
    #[serde(default = "get_true")]
    pub embedding_summation: bool,
    #[serde(default = "get_false")]
    pub sqrt_scaling: bool,
    #[serde(default = "get_default_hashstack_config")]
    pub hash_stack_config: HashStackConfig,
    // index_prefix: different prefix add to index of different features, to prevent bucket conflict for each feature embedding.
    #[serde(default = "get_zero")]
    pub index_prefix: u64,
}

#[derive(Debug, Serialize, Deserialize, Readable, Writable, Clone)]
pub struct EmbeddingConfig {
    #[serde(default = "get_eight")]
    pub feature_index_prefix_bit: usize,
    pub slot_configs: indexmap::IndexMap<String, SlotConfig>,
    #[serde(default = "get_default_feature_groups")]
    pub feature_groups: indexmap::IndexMap<String, Vec<String>>,
}

impl EmbeddingConfig {
    pub fn set(file_path: &PathBuf) -> Result<(), PersiaGlobalConfigError> {
        if !file_path.is_file() {
            tracing::error!("embedding config yaml file NOT found");
            std::thread::sleep(std::time::Duration::from_secs(120));
            panic!("embedding config yaml file NOT found")
        }

        let embedding_config: EmbeddingConfig = serde_yaml::from_reader(
            std::fs::File::open(file_path).expect("cannot read config file"),
        )
        .expect("cannot parse config file");

        let embedding_config = parse_embedding_config(embedding_config);

        tracing::info!("setting embedding_config {:?}", embedding_config,);
        PERSIA_EMBEDDING_CONFIG
            .set(Arc::new(embedding_config))
            .map_err(|_| PersiaGlobalConfigError::SetError)?;

        Ok(())
    }

    pub fn get() -> Result<Arc<Self>, PersiaGlobalConfigError> {
        let singleton = PERSIA_EMBEDDING_CONFIG.get();
        match singleton {
            Some(s) => Ok(s.clone()),
            None => Err(PersiaGlobalConfigError::NotReadyError),
        }
    }
}

pub fn parse_embedding_config(config: EmbeddingConfig) -> EmbeddingConfig {
    let mut config = config;
    let slot_configs = &mut config.slot_configs;
    let feature_groups = &mut config.feature_groups;
    let mut slot_name_to_feature_froup = HashMap::new();

    feature_groups
        .iter()
        .for_each(|(feature_group_name, slot_names)| {
            slot_names.iter().for_each(|slot_name| {
                slot_name_to_feature_froup.insert(slot_name.clone(), feature_group_name.clone());
            })
        });

    slot_configs.iter().for_each(|(slot_name, _)| {
        if !slot_name_to_feature_froup.contains_key(slot_name) {
            let res = feature_groups.insert(slot_name.clone(), vec![slot_name.clone()]);
            if res.is_some() {
                panic!("a slot name can not same with feature group name");
            }
            slot_name_to_feature_froup.insert(slot_name.clone(), slot_name.clone());
        }
    });

    assert_ne!(
        config.feature_index_prefix_bit, 0,
        "feature_index_prefix_bit must > 0"
    );
    let feature_prefix_bias = u64::BITS - config.feature_index_prefix_bit as u32;
    slot_configs
        .iter_mut()
        .for_each(|(slot_name, slot_config)| {
            assert_eq!(
                slot_config.index_prefix, 0,
                "please do not set index_prefix manually"
            );
            let feature_group_name = slot_name_to_feature_froup
                .get(slot_name)
                .expect("feature group not found");
            let feature_froup_index = feature_groups
                .get_index_of(feature_group_name)
                .expect("feature group not found");
            slot_config.index_prefix = num_traits::CheckedShl::checked_shl(
                &(feature_froup_index as u64 + 1),
                feature_prefix_bias,
            )
            .expect("slot index_prefix overflow, please try a bigger feature_index_prefix_bit");
        });

    config
}
