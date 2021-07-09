use crate::monitor::EmbeddingMonitorInner;

use std::sync::Arc;

use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use persia_embedding_config::feature_config::ExtractTargetConfig;
use persia_speedy::{Readable, Writable};

#[allow(dead_code)]
fn get_true() -> bool {
    true
}

fn get_false() -> bool {
    false
}

fn get_zero() -> u64 {
    0
}

fn get_zero_usize() -> usize {
    0
}

fn get_ten() -> usize {
    10
}

fn get_thousand() -> usize {
    1000
}

fn hashstack_config_default() -> HashStackConfig {
    HashStackConfig {
        hash_stack_rounds: 0,
        embedding_size: 0,
    }
}

fn feature_groups_default() -> indexmap::IndexMap<u64, Vec<String>> {
    indexmap::IndexMap::new()
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
    #[serde(default = "get_false")]
    pub is_pid: bool,
    #[serde(default = "get_false")]
    pub is_uid: bool,
    #[serde(default = "hashstack_config_default")]
    pub hash_stack_config: HashStackConfig,
    // index_prefix: different prefix add to index of different features, to prevent bucket conflict for each feature embedding.
    #[serde(default = "get_zero")]
    pub index_prefix: u64,
}

#[derive(Debug, Serialize, Deserialize, Readable, Writable, Clone)]
pub struct EmbeddingConfig {
    /// starts limiting requests at this point
    #[serde(default = "get_thousand")]
    pub forward_buffer_size: usize,
    #[serde(default = "get_zero_usize")]
    pub feature_index_prefix_bit: usize,
    pub slot_config: indexmap::IndexMap<String, SlotConfig>,
    #[serde(default = "feature_groups_default")]
    pub feature_groups: indexmap::IndexMap<u64, Vec<String>>,
    pub target_config: Vec<ExtractTargetConfig>,
}

impl EmbeddingConfig {
    pub fn from_ptr(ptr: *const EmbeddingConfig) -> &'static Self {
        unsafe { &*ptr }
    }
}

#[derive(Clone)]
pub struct FeatureGroup {
    pub group_name: String,
    pub group_prefix: u64,
    pub monitor: Option<Arc<EmbeddingMonitorInner>>,
}

pub fn convert_middleware_config(config: &mut EmbeddingConfig) -> HashMap<String, FeatureGroup> {
    let mut feature2group: HashMap<String, FeatureGroup> = HashMap::new();

    let feature_prefix_bias = u64::BITS - config.feature_index_prefix_bit as u32;

    if config.feature_groups.len() == 0 {
        tracing::info!("feature group NOT set, generating feature group for each feature");
        for (iter, (feature_name, slot)) in config.slot_config.iter_mut().enumerate() {
            if slot.index_prefix > 0 {
                tracing::info!(
                    "index prefix has been set, feature name: {:?}, prefix index: {:?}",
                    feature_name,
                    slot.index_prefix
                );
            } else {
                let prefix_from_iter = iter as u64 + 1;
                slot.index_prefix = prefix_from_iter
                    .checked_shl(feature_prefix_bias)
                    .expect("excessive feature_index_prefix cause overflow");
                config
                    .feature_groups
                    .insert(prefix_from_iter, vec![feature_name.clone()]);
            }
            feature2group.insert(
                feature_name.clone(),
                FeatureGroup {
                    group_name: feature_name.clone(),
                    group_prefix: slot.index_prefix.clone(),
                    monitor: Some(Arc::new(EmbeddingMonitorInner::new(feature_name.clone()))),
                },
            );
        }
    } else {
        let mut seen_prefix: HashSet<u64> = HashSet::new();
        for (group_prefix, features) in config.feature_groups.iter() {
            assert!(!seen_prefix.contains(group_prefix));
            seen_prefix.insert(group_prefix.clone());
            let feature_group = FeatureGroup {
                group_name: group_prefix.to_string(),
                group_prefix: group_prefix.clone(),
                monitor: Some(Arc::new(EmbeddingMonitorInner::new(
                    group_prefix.to_string(),
                ))),
            };
            features.iter().for_each(|feature_name| {
                assert!(
                    !feature2group.contains_key(feature_name),
                    "feature can NOT assign to muti group"
                );
                feature2group.insert(feature_name.clone(), feature_group.clone());
            });
        }
        feature2group.iter().for_each(|(feature_name, feature_group)| {
            let slot_conf = config.slot_config.get_mut(feature_name)
                .expect("feature in feature group NOT found in slot_config");
            if slot_conf.index_prefix == 0 {
                slot_conf.index_prefix = feature_group.group_prefix.checked_shl(feature_prefix_bias)
                    .expect("excessive feature_index_prefix cause overflow");
            }
            else {
                tracing::info!("raw middleware config alreay have index prefix value for slot, you may using a converted yaml");
            }
        });
    }

    feature2group
}
