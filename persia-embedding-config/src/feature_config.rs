use persia_speedy::{Readable, Writable};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SingleFeatureConfig {
    pub name: String,
    pub is_uid: Option<bool>,
    pub is_pid: Option<bool>,
    pub use_embedding_servers: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Readable, Writable, Debug, Clone)]
pub struct ExtractTargetConfig {
    pub name: String,
    pub negative: Vec<String>,
    pub positive: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SimpleTargetConfig {
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TargetConfig {
    SimpleTarget(SimpleTargetConfig),
    ExtractTarget(Vec<ExtractTargetConfig>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FeatureGroupConfig {
    pub targets: TargetConfig,
    pub children: Vec<SingleFeatureConfig>,
}
