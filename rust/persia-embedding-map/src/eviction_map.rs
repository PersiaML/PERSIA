use crate::array_linked_list::{ArrayLinkedList, PersiaArrayLinkedList, PersiaArrayLinkedListImpl};
use crate::emb_entry::{
    ArrayEmbeddingEntry, DynamicEmbeddingEntry, PersiaEmbeddingEntry, PersiaEmbeddingEntryMut,
    PersiaEmbeddingEntryRef,
};
use persia_common::optim::Optimizer;
use persia_embedding_config::{
    EmbeddingConfig, EmbeddingParameterServerConfig, InitializationMethod,
    PersiaEmbeddingModelHyperparameters, PersiaReplicaInfo,
};
use persia_libs::hashbrown::HashMap;
use persia_speedy::{Context, Readable, Writable};
use std::convert::TryFrom;
use std::hash::Hash;

#[derive(Clone, Readable, Writable, Debug)]
pub struct NodeIndex {
    pub linkedlist_index: u32,
    pub array_index: u32,
}

#[derive(Readable, Writable)]
pub struct EvictionMap {
    pub hashmap: HashMap<u64, NodeIndex>,
    pub linkedlists: Vec<PersiaArrayLinkedList>,
    pub embedding_config: EmbeddingConfig,
    pub hyperparameters: PersiaEmbeddingModelHyperparameters,
}

impl EvictionMap {
    pub fn new(
        embedding_config: &EmbeddingConfig,
        embedding_parameter_server_config: &EmbeddingParameterServerConfig,
        replica_info: &PersiaReplicaInfo,
        optimizer_space: usize,
        hyperparameters: PersiaEmbeddingModelHyperparameters,
    ) -> Self {
        let bucket_size = embedding_parameter_server_config.num_hashmap_internal_shards;
        let replica_size = replica_info.replica_size;

        let linkedlists = embedding_config
            .slots_config
            .iter()
            .map(|(_, slot_config)| {
                let capacity = slot_config.capacity / bucket_size / replica_size;
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
                linkedlist
            })
            .collect();

        Self {
            hashmap: HashMap::new(),
            linkedlists,
            embedding_config: embedding_config.clone(),
            hyperparameters: hyperparameters.clone(),
        }
    }

    pub fn get(&self, key: &u64) -> Option<PersiaEmbeddingEntryRef> {
        match self.hashmap.get(key) {
            Some(idx) => self.linkedlists[idx.linkedlist_index as usize].get(idx.array_index),
            None => None,
        }
    }

    pub fn get_dyn(&self, key: &u64) -> Option<DynamicEmbeddingEntry> {
        match self.hashmap.get(key) {
            Some(idx) => {
                let entry_ref = self.linkedlists[idx.linkedlist_index as usize]
                    .get(idx.array_index)
                    .unwrap();
                let entry_dyn = DynamicEmbeddingEntry {
                    inner: entry_ref.inner.to_vec(),
                    embedding_dim: entry_ref.embedding_dim,
                    sign: entry_ref.sign,
                    slot_index: idx.linkedlist_index as usize,
                };
                Some(entry_dyn)
            }
            None => None,
        }
    }

    pub fn get_mut(&mut self, key: &u64) -> Option<PersiaEmbeddingEntryMut> {
        match self.hashmap.get(key) {
            Some(idx) => self.linkedlists[idx.linkedlist_index as usize].get_mut(idx.array_index),
            None => None,
        }
    }

    pub fn get_refresh(&mut self, key: &u64) -> Option<PersiaEmbeddingEntryRef> {
        match self.hashmap.get_mut(key) {
            Some(idx) => {
                let new_idx =
                    self.linkedlists[idx.linkedlist_index as usize].move_to_back(idx.array_index);
                idx.array_index = new_idx;
                self.linkedlists[idx.linkedlist_index as usize].back()
            }
            None => None,
        }
    }

    pub fn insert_dyn(&mut self, key: u64, value: DynamicEmbeddingEntry) {
        if let Some(idx) = self.hashmap.get_mut(&key) {
            self.linkedlists[idx.linkedlist_index as usize].remove(idx.array_index);
            let new_idx = self.linkedlists[idx.linkedlist_index as usize].push_back(value);
            idx.array_index = new_idx;
        } else {
            let linkedlist_index = value.slot_index;
            let array_index = self.linkedlists[linkedlist_index].push_back(value);
            self.hashmap.insert(
                key.clone(),
                NodeIndex {
                    linkedlist_index: linkedlist_index as u32,
                    array_index,
                },
            );

            self.evict(linkedlist_index);
        }
    }

    pub fn insert_init(
        &mut self,
        key: u64,
        optimizer_space: usize,
        slot_name: &String,
    ) -> PersiaEmbeddingEntryMut {
        if self.hashmap.get(&key).is_none() {
            let linkedlist_index = self.embedding_config.get_index_by_name(slot_name);
            let slot_config = self.embedding_config.get_slot_by_name(slot_name);

            let array_index = self.linkedlists[linkedlist_index].push_back_init(
                &self.hyperparameters.initialization_method,
                slot_config.dim,
                optimizer_space,
                key,
                key,
            );
            self.hashmap.insert(
                key,
                NodeIndex {
                    linkedlist_index: linkedlist_index as u32,
                    array_index,
                },
            );

            self.linkedlists[linkedlist_index]
                .get_mut(array_index)
                .unwrap()
        } else {
            self.get_mut(&key).unwrap()
        }
    }

    pub fn clear(&mut self) {
        self.hashmap.clear();
        self.linkedlists.iter_mut().for_each(|list| list.clear());
    }

    pub fn len(&self) -> usize {
        self.hashmap.len()
    }

    fn evict(&mut self, linkedlist_index: usize) {
        if self.linkedlists[linkedlist_index].len()
            > self
                .embedding_config
                .slots_config
                .get_index(linkedlist_index)
                .unwrap()
                .1
                .capacity
        {
            if let Some(evicted) = self.linkedlists[linkedlist_index].pop_front() {
                self.hashmap.remove(&evicted);
            }
        }
    }
}

#[cfg(test)]
mod eviction_map_tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use persia_embedding_config::{get_default_hashstack_config, SlotConfig};
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
        let optimizer_space: usize = 0;
        let mut map = EvictionMap::new(&embedding_config, optimizer_space);

        for i in 0..4 {
            let entry = DynamicEmbeddingEntry {
                inner: vec![0.0_f32; 8],
                embedding_dim: 8,
                sign: i,
                slot_index: 0,
            };
            map.insert_dyn(&i, entry);
        }

        assert_eq!(map.len(), 4);

        for i in 10..18 {
            let entry = DynamicEmbeddingEntry {
                inner: vec![0.0_f32; 16],
                embedding_dim: 16,
                sign: i,
                slot_index: 1,
            };
            map.insert_dyn(&i, entry);
        }

        assert_eq!(map.len(), 12);

        for i in 4..8 {
            let entry = DynamicEmbeddingEntry {
                inner: vec![0.0_f32; 8],
                embedding_dim: 8,
                sign: i,
                slot_index: 0,
            };
            map.insert_dyn(&i, entry);
        }

        assert_eq!(map.len(), 12);
        assert_eq!(map.get_refresh(&3).is_none(), true);
        assert_eq!(map.get_refresh(&4).is_some(), true);

        map.insert_dyn(
            &8,
            DynamicEmbeddingEntry {
                inner: vec![0.0_f32; 8],
                embedding_dim: 8,
                sign: 8,
                slot_index: 0,
            },
        );

        assert_eq!(map.len(), 12);
        assert_eq!(map.get(&5).is_none(), true);
        assert_eq!(map.get(&6).is_some(), true);
    }
}
