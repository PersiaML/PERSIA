#![allow(clippy::needless_return)]

pub mod feature_config;
use persia_speedy::{Readable, Writable};
use serde::{Deserialize, Serialize};
use std::{io::Read, path::PathBuf, sync::Arc, u64};
use thiserror::Error;
use yaml_rust::YamlLoader;

#[derive(Error, Debug, Clone)]
pub enum PersiaGlobalConfigError {
    #[error("global config not ready error")]
    NotReadyError,
    #[error("global config set error")]
    SetError,
    #[error("failed to open/read config file error")]
    ConfigFileError,
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

static PERSIA_GLOBAL_CONFIG: once_cell::sync::OnceCell<
    Arc<parking_lot::RwLock<PersiaGlobalConfig>>,
> = once_cell::sync::OnceCell::new();

#[derive(Readable, Writable, Debug, Clone)]
pub enum PerisaShardedServerIntent {
    Train,
    Eval,
    Infer,
}

#[derive(Readable, Writable, Debug, Clone)]
pub enum PersiaPersistenceStorage {
    Ceph,
    Hdfs,
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct PersiaShardedServerConfig {
    // cur task
    pub intent: PerisaShardedServerIntent,

    // Eviction map config
    pub capacity: usize,
    pub num_hashmap_internal_shards: usize,
    pub full_amount_manager_buffer_size: usize,

    // model persistence config
    pub num_persistence_workers: usize,
    pub num_signs_per_file: usize,
    pub storage: PersiaPersistenceStorage,
    pub local_buffer_dir: String,

    // incremental dump config
    pub enable_incremental_update: bool,
    pub incremental_buffer_size: usize,
    pub incremental_dir: String,
    pub incremental_channel_capacity: usize,
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct PersiaReplicaInfo {
    pub replica_name: String,
    pub replica_size: usize,
    pub replica_index: usize,
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct PersiaMetricsConfig {
    pub enable_metrics: bool,
    pub job_name: String,
    pub instance_name: String,
    pub ip_addr: String,
    pub push_interval: u64,
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct PersiaSparseModelHyperparameters {
    pub initialization_method: InitializationMethod,
    pub admit_probability: f32,
    pub weight_bound: f32,
    pub enable_weight_bound: bool,
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct PersiaMiddlewareConfig {
    pub forward_buffer_size: usize,
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct PersiaGlobalConfig {
    pub replica_info: PersiaReplicaInfo,
    pub sharded_server_config: PersiaShardedServerConfig,
    pub middleware_config: PersiaMiddlewareConfig,
    pub metrics_config: PersiaMetricsConfig,
}

impl PersiaGlobalConfig {
    pub fn get() -> Result<Arc<parking_lot::RwLock<Self>>, PersiaGlobalConfigError> {
        let singleton = PERSIA_GLOBAL_CONFIG.get();
        match singleton {
            Some(s) => Ok(s.clone()),
            None => Err(PersiaGlobalConfigError::NotReadyError),
        }
    }

    pub fn set(
        config_file: PathBuf,
        replica_index: usize,
        replica_size: usize,
        replica_name: String,
    ) -> Result<(), PersiaGlobalConfigError> {
        let f = std::fs::File::open(config_file);
        if f.is_err() {
            tracing::error!("failed to open config file, {:?}", f.unwrap_err());
            return Err(PersiaGlobalConfigError::ConfigFileError);
        }
        let mut f = f.unwrap();
        let mut content = String::new();
        let res = f.read_to_string(&mut content);
        if res.is_err() {
            tracing::error!("failed to read config file, {:?}", res.unwrap_err());
            return Err(PersiaGlobalConfigError::ConfigFileError);
        }

        let yaml = YamlLoader::load_from_str(&content);
        if yaml.is_err() {
            tracing::error!("fail to parse config file");
            return Err(PersiaGlobalConfigError::ConfigFileError);
        }
        let yaml = yaml.unwrap();
        let conf = yaml.first().unwrap();

        let server_intent = conf["PersiaShardedServerConfig"]["intent"]
            .as_str()
            .unwrap_or("train");
        let intent = {
            if server_intent == "train" {
                PerisaShardedServerIntent::Train
            } else if server_intent == "infer" {
                PerisaShardedServerIntent::Infer
            } else if server_intent == "eval" {
                PerisaShardedServerIntent::Eval
            } else {
                panic!("unknown intent, it must be one of train/infer/eval");
            }
        };
        let capacity = conf["PersiaShardedServerConfig"]["capacity"]
            .as_i64()
            .unwrap_or(100000000) as usize;
        let num_hashmap_internal_shards = conf["PersiaShardedServerConfig"]
            ["num_hashmap_internal_shards"]
            .as_i64()
            .unwrap_or(128) as usize;
        let full_amount_manager_buffer_size = conf["PersiaShardedServerConfig"]
            ["full_amount_manager_buffer_size"]
            .as_i64()
            .unwrap_or(1000) as usize;
        let num_persistence_workers = conf["PersiaShardedServerConfig"]["num_persistence_workers"]
            .as_i64()
            .unwrap_or(4) as usize;
        let num_signs_per_file = conf["PersiaShardedServerConfig"]["num_signs_per_file"]
            .as_i64()
            .unwrap_or(5000000) as usize;
        let storage = {
            let storage_str = conf["PersiaShardedServerConfig"]["storage"]
                .as_str()
                .unwrap_or("hdfs");
            if storage_str == "hdfs" {
                PersiaPersistenceStorage::Hdfs
            } else if storage_str == "ceph" {
                PersiaPersistenceStorage::Ceph
            } else {
                panic!("unknown storage, it must be one of hdfs/ceph");
            }
        };
        let enable_incremental_update = conf["PersiaShardedServerConfig"]
            ["enable_incremental_update"]
            .as_bool()
            .unwrap_or(false);
        let local_buffer_dir = String::from(
            conf["PersiaShardedServerConfig"]["local_buffer_dir"]
                .as_str()
                .unwrap_or("/tmp/"),
        );
        let incremental_buffer_size = conf["PersiaShardedServerConfig"]["incremental_buffer_size"]
            .as_i64()
            .unwrap_or(5000000) as usize;
        let incremental_dir = String::from(
            conf["PersiaShardedServerConfig"]["incremental_dir"]
                .as_str()
                .unwrap_or("/tmp/persia_inc/"),
        );
        let incremental_channel_capacity = conf["PersiaShardedServerConfig"]
            ["incremental_channel_capacity"]
            .as_i64()
            .unwrap_or(1000) as usize;

        let forward_buffer_size = conf["PersiaMiddlewareConfig"]["forward_buffer_size"]
            .as_i64()
            .unwrap_or(1000) as usize;

        let enable_metrics = conf["PersiaMetricsConfig"]["enable_metrics"]
            .as_bool()
            .unwrap_or(false);
        let push_interval = conf["PersiaMetricsConfig"]["push_interval"]
            .as_i64()
            .unwrap_or(10) as u64;
        let job_name =
            std::env::var("PERSIA_JOB_NAME").unwrap_or(String::from("persia_default_job_name"));
        let instance_name = format!("{}-{}", replica_name, replica_size);
        let ip_addr = local_ipaddress::get().unwrap_or(String::from("NA"));

        let singleton = Self {
            replica_info: PersiaReplicaInfo {
                replica_name,
                replica_size,
                replica_index,
            },
            sharded_server_config: PersiaShardedServerConfig {
                intent,
                capacity,
                num_hashmap_internal_shards,
                full_amount_manager_buffer_size,
                num_persistence_workers,
                num_signs_per_file,
                storage,
                local_buffer_dir,
                enable_incremental_update,
                incremental_buffer_size,
                incremental_dir,
                incremental_channel_capacity,
            },
            middleware_config: PersiaMiddlewareConfig {
                forward_buffer_size,
            },
            metrics_config: PersiaMetricsConfig {
                enable_metrics,
                job_name,
                instance_name,
                ip_addr,
                push_interval,
            },
        };

        tracing::info!("PersiaGlobalConfig parsed \n{:?}", singleton);

        let singleton = Arc::new(parking_lot::RwLock::new(singleton));
        let res = PERSIA_GLOBAL_CONFIG.set(singleton);
        if res.is_err() {
            return Err(PersiaGlobalConfigError::SetError);
        }

        Ok(())
    }
}
