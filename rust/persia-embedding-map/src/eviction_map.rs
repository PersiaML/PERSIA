use crate::array_linked_list::{ArrayLinkedList, PersiaArrayLinkedList, PersiaArrayLinkedListImpl};
use crate::emb_entry::{
    ArrayEmbeddingEntry, DynamicEmbeddingEntry, PersiaEmbeddingEntryMut, PersiaEmbeddingEntryRef,
};
use persia_common::optim::Optimizable;
use persia_embedding_config::{
    EmbeddinHyperparameters, EmbeddingConfig, EmbeddingParameterServerConfig, PersiaReplicaInfo,
    SlotConfig,
};
use persia_libs::hashbrown::HashMap;
use persia_speedy::{Readable, Writable};
use std::sync::Arc;

#[derive(Readable, Writable)]
pub struct LruCache {
    pub hashmap: HashMap<u64, u32>,
    pub linkedlist: PersiaArrayLinkedList,
    pub slot_index: usize,
    pub capacity: usize,
}

impl LruCache {
    pub fn get(&self, sign: &u64) -> Option<PersiaEmbeddingEntryRef> {
        match self.hashmap.get(sign) {
            Some(idx) => self.linkedlist.get(*idx),
            None => None,
        }
    }

    pub fn get_dyn(&self, sign: &u64) -> Option<DynamicEmbeddingEntry> {
        match self.hashmap.get(sign) {
            Some(idx) => {
                let entry_ref: PersiaEmbeddingEntryRef = self.linkedlist.get(*idx).unwrap();
                let entry_dyn = DynamicEmbeddingEntry {
                    inner: entry_ref.inner.to_vec(),
                    embedding_dim: entry_ref.embedding_dim,
                    sign: entry_ref.sign,
                    slot_index: self.slot_index,
                };
                Some(entry_dyn)
            }
            None => None,
        }
    }

    pub fn get_mut(&mut self, sign: &u64) -> Option<PersiaEmbeddingEntryMut> {
        match self.hashmap.get(sign) {
            Some(idx) => self.linkedlist.get_mut(*idx),
            None => None,
        }
    }

    pub fn get_refresh(&mut self, sign: &u64) -> Option<PersiaEmbeddingEntryRef> {
        match self.hashmap.get_mut(sign) {
            Some(idx) => {
                let new_idx = self.linkedlist.move_to_back(*idx);
                *idx = new_idx;
                self.linkedlist.back()
            }
            None => None,
        }
    }

    pub fn insert_dyn(&mut self, sign: u64, value: DynamicEmbeddingEntry) {
        if let Some(idx) = self.hashmap.get_mut(&sign) {
            self.linkedlist.remove(*idx);
            let new_idx = self.linkedlist.push_back(value);
            *idx = new_idx;
        } else {
            let idx = self.linkedlist.push_back(value);
            self.hashmap.insert(sign, idx);
            self.evict();
        }
    }

    pub fn insert_init(
        &mut self,
        sign: u64,
        optimizer: Arc<Box<dyn Optimizable + Send + Sync>>,
        slot_config: &SlotConfig,
        hyperparameters: &EmbeddinHyperparameters,
    ) -> PersiaEmbeddingEntryRef {
        if self.hashmap.get(&sign).is_none() {
            let idx = self.linkedlist.push_back_init(
                &hyperparameters.initialization_method,
                slot_config.dim,
                optimizer,
                sign,
                sign,
            );
            self.hashmap.insert(sign, idx);

            self.evict();

            self.linkedlist.get(idx).unwrap()
        } else {
            self.get(&sign).unwrap()
        }
    }

    pub fn clear(&mut self) {
        self.hashmap.clear();
        self.linkedlist.clear();
    }

    pub fn len(&self) -> usize {
        self.hashmap.len()
    }

    fn evict(&mut self) {
        if self.linkedlist.len() > self.capacity {
            if let Some(evicted) = self.linkedlist.pop_front() {
                self.hashmap.remove(&evicted);
            }
        }
    }
}

#[derive(Readable, Writable)]
pub struct EvictionMap {
    pub lru_caches: Vec<LruCache>,
    pub embedding_config: EmbeddingConfig,
    pub hyperparameters: EmbeddinHyperparameters,
    pub shard_idx: usize,
}

impl Default for EvictionMap {
    fn default() -> Self {
        Self {
            lru_caches: Vec::new(),
            embedding_config: EmbeddingConfig::default(),
            hyperparameters: EmbeddinHyperparameters::default(),
            shard_idx: 0,
        }
    }
}

impl EvictionMap {
    pub fn new(
        embedding_config: &EmbeddingConfig,
        embedding_parameter_server_config: &EmbeddingParameterServerConfig,
        replica_info: &PersiaReplicaInfo,
        optimizer: Arc<Box<dyn Optimizable + Send + Sync>>,
        hyperparameters: EmbeddinHyperparameters,
        shard_idx: usize,
    ) -> Self {
        let bucket_size = embedding_parameter_server_config.num_hashmap_internal_shards;
        let replica_size = replica_info.replica_size;

        let lru_caches = embedding_config
            .slots_config
            .iter()
            .enumerate()
            .map(|(slot_index, (_, slot_config))| {
                let capacity = slot_config.capacity / bucket_size / replica_size;
                let optimizer_space = optimizer.require_space(slot_config.dim);
                let linkedlist: PersiaArrayLinkedList = match slot_config.dim + optimizer_space {
                    1 => {
                        PersiaArrayLinkedList::Array1(
                            ArrayLinkedList::<ArrayEmbeddingEntry<f32, 1>>::with_capacity(capacity),
                        )
                    }
                    2 => {
                        PersiaArrayLinkedList::Array2(
                            ArrayLinkedList::<ArrayEmbeddingEntry<f32, 2>>::with_capacity(capacity),
                        )
                    }
                    4 => {
                        PersiaArrayLinkedList::Array4(
                            ArrayLinkedList::<ArrayEmbeddingEntry<f32, 4>>::with_capacity(capacity),
                        )
                    }
                    8 => {
                        PersiaArrayLinkedList::Array8(
                            ArrayLinkedList::<ArrayEmbeddingEntry<f32, 8>>::with_capacity(capacity),
                        )
                    }
                    12 => PersiaArrayLinkedList::Array12(ArrayLinkedList::<
                        ArrayEmbeddingEntry<f32, 12>,
                    >::with_capacity(
                        capacity
                    )),
                    16 => PersiaArrayLinkedList::Array16(ArrayLinkedList::<
                        ArrayEmbeddingEntry<f32, 16>,
                    >::with_capacity(
                        capacity
                    )),
                    24 => PersiaArrayLinkedList::Array24(ArrayLinkedList::<
                        ArrayEmbeddingEntry<f32, 24>,
                    >::with_capacity(
                        capacity
                    )),
                    32 => PersiaArrayLinkedList::Array32(ArrayLinkedList::<
                        ArrayEmbeddingEntry<f32, 32>,
                    >::with_capacity(
                        capacity
                    )),
                    48 => PersiaArrayLinkedList::Array48(ArrayLinkedList::<
                        ArrayEmbeddingEntry<f32, 48>,
                    >::with_capacity(
                        capacity
                    )),
                    64 => PersiaArrayLinkedList::Array64(ArrayLinkedList::<
                        ArrayEmbeddingEntry<f32, 64>,
                    >::with_capacity(
                        capacity
                    )),
                    96 => PersiaArrayLinkedList::Array96(ArrayLinkedList::<
                        ArrayEmbeddingEntry<f32, 96>,
                    >::with_capacity(
                        capacity
                    )),
                    128 => PersiaArrayLinkedList::Array128(ArrayLinkedList::<
                        ArrayEmbeddingEntry<f32, 128>,
                    >::with_capacity(
                        capacity
                    )),
                    _ => PersiaArrayLinkedList::ArrayDyn(ArrayLinkedList::<
                        ArrayEmbeddingEntry<f32, 0>,
                    >::with_capacity(
                        capacity
                    )),
                };
                LruCache {
                    linkedlist,
                    hashmap: HashMap::with_capacity(capacity),
                    slot_index,
                    capacity,
                }
            })
            .collect();

        Self {
            lru_caches,
            embedding_config: embedding_config.clone(),
            hyperparameters: hyperparameters.clone(),
            shard_idx,
        }
    }

    pub fn get(&self, sign: &u64, slot_index: &usize) -> Option<PersiaEmbeddingEntryRef> {
        self.lru_caches[*slot_index].get(sign)
    }

    pub fn get_dyn(&self, sign: &u64, slot_index: &usize) -> Option<DynamicEmbeddingEntry> {
        self.lru_caches[*slot_index].get_dyn(sign)
    }

    pub fn get_mut(&mut self, sign: &u64, slot_index: &usize) -> Option<PersiaEmbeddingEntryMut> {
        self.lru_caches[*slot_index].get_mut(sign)
    }

    pub fn get_refresh(
        &mut self,
        sign: &u64,
        slot_index: &usize,
    ) -> Option<PersiaEmbeddingEntryRef> {
        self.lru_caches[*slot_index].get_refresh(sign)
    }

    pub fn insert_dyn(&mut self, sign: u64, value: DynamicEmbeddingEntry) {
        let slot_index = value.slot_index;
        self.lru_caches[slot_index].insert_dyn(sign, value)
    }

    pub fn insert_init(
        &mut self,
        sign: u64,
        slot_index: &usize,
        optimizer: Arc<Box<dyn Optimizable + Send + Sync>>,
    ) -> PersiaEmbeddingEntryRef {
        let slot_config = self.embedding_config.get_slot_by_index(*slot_index);
        self.lru_caches[*slot_index].insert_init(
            sign,
            optimizer,
            slot_config,
            &self.hyperparameters,
        )
    }

    pub fn clear(&mut self) {
        self.lru_caches.iter_mut().for_each(|lru| lru.clear());
    }

    pub fn len(&self) -> usize {
        self.lru_caches.iter().map(|lru| lru.len()).sum::<usize>()
    }
}

#[cfg(test)]
mod eviction_map_tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use persia_common::optim::{NaiveSGDConfig, Optimizer, OptimizerConfig};
    use persia_embedding_config::{
        get_default_hashstack_config, EmbeddingParameterServerConfig, PersiaReplicaInfo, SlotConfig,
    };
    use persia_libs::indexmap::indexmap;

    #[test]
    fn test_evict() {
        let embedding_config = EmbeddingConfig {
            slots_config: indexmap! {
                "first_slot".to_string() => SlotConfig {
                    dim: 8,
                    capacity: 4,
                    sample_fixed_size: 0,
                    embedding_summation: true,
                    sqrt_scaling: false,
                    hash_stack_config: get_default_hashstack_config(),
                },
                "second_slot".to_string() => SlotConfig {
                    dim: 16,
                    capacity: 8,
                    sample_fixed_size: 0,
                    embedding_summation: true,
                    sqrt_scaling: false,
                    hash_stack_config: get_default_hashstack_config(),
                },
            },
        };

        let mut embedding_parameter_server_config = EmbeddingParameterServerConfig::default();
        embedding_parameter_server_config.num_hashmap_internal_shards = 1;

        let replica_info = PersiaReplicaInfo {
            replica_index: 0,
            replica_size: 1,
        };

        let optimizer = Optimizer::new(OptimizerConfig::SGD(NaiveSGDConfig { lr: 0.01, wd: 0.01 }));
        let optimizer = Arc::new(optimizer.to_optimizable());
        let hyperparameters = EmbeddinHyperparameters::default();

        let mut map = EvictionMap::new(
            &embedding_config,
            &embedding_parameter_server_config,
            &replica_info,
            optimizer,
            hyperparameters,
            0,
        );

        for i in 0..4 {
            let entry = DynamicEmbeddingEntry {
                inner: vec![0.0_f32; 8],
                embedding_dim: 8,
                sign: i,
                slot_index: 0,
            };
            map.insert_dyn(i, entry);
        }

        assert_eq!(map.len(), 4);

        for i in 10..18 {
            let entry = DynamicEmbeddingEntry {
                inner: vec![0.0_f32; 16],
                embedding_dim: 16,
                sign: i,
                slot_index: 1,
            };
            map.insert_dyn(i, entry);
        }

        assert_eq!(map.len(), 12);

        for i in 4..8 {
            let entry = DynamicEmbeddingEntry {
                inner: vec![0.0_f32; 8],
                embedding_dim: 8,
                sign: i,
                slot_index: 0,
            };
            map.insert_dyn(i, entry);
        }

        assert_eq!(map.len(), 12);
        assert_eq!(map.get_refresh(&3, &0).is_none(), true);
        assert_eq!(map.get_refresh(&4, &0).is_some(), true);

        map.insert_dyn(
            8,
            DynamicEmbeddingEntry {
                inner: vec![0.0_f32; 8],
                embedding_dim: 8,
                sign: 8,
                slot_index: 0,
            },
        );

        assert_eq!(map.len(), 12);
        assert_eq!(map.get(&5, &0).is_none(), true);
        assert_eq!(map.get(&6, &0).is_some(), true);
    }
}
