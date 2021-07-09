#![allow(clippy::needless_return)]

pub mod feature_config;
use once_cell::sync::OnceCell;
use persia_speedy::{Readable, Writable};
use serde::{Deserialize, Serialize};
use std::{io::Read, path::PathBuf, sync::Arc};
use thiserror::Error;
use yaml_rust::{Yaml, YamlLoader};

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

static PERSIA_EMBEDDING_SEVER_CONFIG: OnceCell<
    Arc<parking_lot::RwLock<PersiaShardedServerConfig>>,
> = OnceCell::new();

static PERSIA_MIDDLEWARE_CONFIG: OnceCell<Arc<parking_lot::RwLock<PersiaMiddlewareConfig>>> =
    OnceCell::new();

static PERSIA_COMMON_CONFIG: OnceCell<Arc<parking_lot::RwLock<PersiaCommonConfig>>> =
    OnceCell::new();

fn read_yaml(config_file: &PathBuf) -> Result<Yaml, PersiaGlobalConfigError> {
    let f = std::fs::File::open(config_file);
    if f.is_err() {
        let err_msg = format!("{:?}", f.unwrap_err());
        return Err(PersiaGlobalConfigError::ConfigFileError(err_msg));
    }
    let mut f = f.unwrap();
    let mut content = String::new();
    let res = f.read_to_string(&mut content);
    if res.is_err() {
        let err_msg = format!("{:?}", res.unwrap_err());
        return Err(PersiaGlobalConfigError::ConfigFileError(err_msg));
    }

    let yaml = YamlLoader::load_from_str(&content);
    if yaml.is_err() {
        let err_msg = format!("{:?}", yaml.unwrap_err());
        return Err(PersiaGlobalConfigError::ConfigFileError(err_msg));
    }
    let yaml = yaml.unwrap();
    let conf = yaml.into_iter().next();
    if conf.is_none() {
        let err_msg = String::from("yaml file not include a yaml");
        return Err(PersiaGlobalConfigError::ConfigFileError(err_msg));
    }
    Ok(conf.unwrap())
}

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
pub struct PersiaReplicaInfo {
    pub replica_size: usize,
    pub replica_index: usize,
}

#[derive(Readable, Writable, Debug, Clone)]
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
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct PersiaShardedServerConfig {
    pub instance_info: InstanceInfo,

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

impl PersiaShardedServerConfig {
    pub fn get() -> Result<Arc<parking_lot::RwLock<Self>>, PersiaGlobalConfigError> {
        let singleton = PERSIA_EMBEDDING_SEVER_CONFIG.get();
        match singleton {
            Some(s) => Ok(s.clone()),
            None => Err(PersiaGlobalConfigError::NotReadyError),
        }
    }

    pub fn set(config_file: &PathBuf, port: u16) -> Result<(), PersiaGlobalConfigError> {
        let conf = read_yaml(config_file)?;

        let instance_info = InstanceInfo::new(port);

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

        let singleton = Self {
            instance_info,
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
        };

        tracing::info!("PersiaShardedServerConfig parsed \n{:?}", singleton);

        let singleton = Arc::new(parking_lot::RwLock::new(singleton));
        let res = PERSIA_EMBEDDING_SEVER_CONFIG.set(singleton);
        if res.is_err() {
            return Err(PersiaGlobalConfigError::SetError);
        }

        Ok(())
    }
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct PersiaMiddlewareConfig {
    pub instance_info: InstanceInfo,
    pub forward_buffer_size: usize,
}

impl PersiaMiddlewareConfig {
    pub fn get() -> Result<Arc<parking_lot::RwLock<Self>>, PersiaGlobalConfigError> {
        let singleton = PERSIA_MIDDLEWARE_CONFIG.get();
        match singleton {
            Some(s) => Ok(s.clone()),
            None => Err(PersiaGlobalConfigError::NotReadyError),
        }
    }

    pub fn set(config_file: &PathBuf, port: u16) -> Result<(), PersiaGlobalConfigError> {
        let conf = read_yaml(config_file)?;

        let instance_info = InstanceInfo::new(port);

        let forward_buffer_size = conf["PersiaMiddlewareConfig"]["forward_buffer_size"]
            .as_i64()
            .unwrap_or(1000) as usize;

        let singleton = Self {
            instance_info,
            forward_buffer_size,
        };

        tracing::info!("PersiaMiddlewareConfig parsed \n{:?}", singleton);
        let singleton = Arc::new(parking_lot::RwLock::new(singleton));
        let res = PERSIA_MIDDLEWARE_CONFIG.set(singleton);
        if res.is_err() {
            return Err(PersiaGlobalConfigError::SetError);
        }
        Ok(())
    }
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
pub struct PersiaCommonConfig {
    pub replica_info: PersiaReplicaInfo,
    pub metrics_config: PersiaMetricsConfig,
}

impl PersiaCommonConfig {
    pub fn get() -> Result<Arc<parking_lot::RwLock<Self>>, PersiaGlobalConfigError> {
        let singleton = PERSIA_COMMON_CONFIG.get();
        match singleton {
            Some(s) => Ok(s.clone()),
            None => Err(PersiaGlobalConfigError::NotReadyError),
        }
    }

    pub fn set(
        config_file: &PathBuf,
        replica_index: usize,
        replica_size: usize,
    ) -> Result<PersiaReplicaInfo, PersiaGlobalConfigError> {
        let conf = read_yaml(config_file)?;

        let replica_info = PersiaReplicaInfo {
            replica_index,
            replica_size,
        };

        let enable_metrics = conf["PersiaMetricsConfig"]["enable_metrics"]
            .as_bool()
            .unwrap_or(false);
        let push_interval = conf["PersiaMetricsConfig"]["push_interval"]
            .as_i64()
            .unwrap_or(10) as u64;
        let job_name =
            std::env::var("PERSIA_JOB_NAME").unwrap_or(String::from("persia_default_job_name"));
        let instance_name = format!("{}", replica_index);
        let ip_addr = get_local_ip();

        let metrics_config = PersiaMetricsConfig {
            enable_metrics,
            job_name,
            instance_name,
            ip_addr,
            push_interval,
        };

        let singleton = Self {
            replica_info: replica_info.clone(),
            metrics_config,
        };

        tracing::info!("PersiaCommonConfig parsed \n{:?}", singleton);
        let singleton = Arc::new(parking_lot::RwLock::new(singleton));
        let res = PERSIA_COMMON_CONFIG.set(singleton);
        if res.is_err() {
            return Err(PersiaGlobalConfigError::SetError);
        }
        Ok(replica_info)
    }
}
