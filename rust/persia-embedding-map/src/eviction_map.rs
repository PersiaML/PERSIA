use crate::array_linked_list::{ArrayLinkedList, PersiaArrayLinkedList};
use crate::emb_entry::{
    ArrayEmbeddingEntry, DynamicEmbeddingEntry, PersiaEmbeddingEntry, PersiaEmbeddingEntryMut,
    PersiaEmbeddingEntryRef,
};
use persia_common::optim::Optimizer;
use persia_embedding_config::{
    EmbeddingConfig, EmbeddingParameterServerConfig, InitializationMethod, PersiaReplicaInfo,
};
use persia_libs::hashbrown::HashMap;
use std::convert::TryFrom;
use std::hash::Hash;

pub struct NodeIndex {
    pub linkedlist_index: u32,
    pub array_index: u32,
}

pub struct EvictionMap {
    pub hashmap: HashMap<u64, NodeIndex>,
    pub linkedlists: Vec<Box<dyn PersiaArrayLinkedList + Send>>,
    pub embedding_config: EmbeddingConfig,
}

impl EvictionMap {
    pub fn new(
        embedding_config: &EmbeddingConfig,
        embedding_parameter_server_config: &EmbeddingParameterServerConfig,
        replica_info: &PersiaReplicaInfo,
        optimizer_space: usize,
    ) -> Self {
        let bucket_size = embedding_parameter_server_config.num_hashmap_internal_shards;
        let replica_size = replica_info.replica_size;

        let linkedlists = embedding_config
            .slots_config
            .iter()
            .map(|(_, slot_config)| {
                let capacity = slot_config.capacity / bucket_size / replica_size;
                let linkedlist: Box<dyn PersiaArrayLinkedList + Send> = match slot_config.dim
                    + optimizer_space
                {
                    1 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 1>>::with_capacity(capacity),
                    ),
                    2 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 2>>::with_capacity(capacity),
                    ),
                    4 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 4>>::with_capacity(capacity),
                    ),
                    8 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 8>>::with_capacity(capacity),
                    ),
                    12 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 12>>::with_capacity(capacity),
                    ),
                    16 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 16>>::with_capacity(capacity),
                    ),
                    24 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 24>>::with_capacity(capacity),
                    ),
                    32 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 32>>::with_capacity(capacity),
                    ),
                    48 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 48>>::with_capacity(capacity),
                    ),
                    64 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 64>>::with_capacity(capacity),
                    ),
                    96 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 96>>::with_capacity(capacity),
                    ),
                    128 => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 128>>::with_capacity(capacity),
                    ),
                    _ => Box::new(
                        ArrayLinkedList::<ArrayEmbeddingEntry<f32, 0>>::with_capacity(capacity),
                    ),
                };
                linkedlist
            })
            .collect();

        Self {
            hashmap: HashMap::new(),
            linkedlists,
            embedding_config: embedding_config.clone(),
        }
    }

    pub fn get(&self, key: &u64) -> Option<PersiaEmbeddingEntryRef> {
        match self.hashmap.get(key) {
            Some(idx) => self.linkedlists[idx.linkedlist_index as usize].get(idx.array_index),
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

    pub fn insert(&mut self, key: u64, value: DynamicEmbeddingEntry, slot_name: &String) {
        if let Some(idx) = self.hashmap.get_mut(&key) {
            self.linkedlists[idx.linkedlist_index as usize].remove(idx.array_index);
            let new_idx = self.linkedlists[idx.linkedlist_index as usize].push_back(value);
            idx.array_index = new_idx;
        } else {
            let linkedlist_index = self
                .embedding_config
                .slots_config
                .get_index_of(slot_name)
                .unwrap();
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
        initialization_method: &InitializationMethod,
        embedding_space: usize,
        optimizer_space: usize,
        slot_name: &String,
    ) {
        if self.hashmap.get(&key).is_none() {
            let linkedlist_index = self
                .embedding_config
                .slots_config
                .get_index_of(slot_name)
                .unwrap();
            let array_index = self.linkedlists[linkedlist_index].push_back_init(
                initialization_method,
                embedding_space,
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
            };
            map.insert(&i, entry, &String::from("first_slot"));
        }

        assert_eq!(map.len(), 4);

        for i in 10..18 {
            let entry = DynamicEmbeddingEntry {
                inner: vec![0.0_f32; 8],
                embedding_dim: 8,
                sign: i,
            };
            map.insert(&i, entry, &String::from("second_slot"));
        }

        assert_eq!(map.len(), 12);

        for i in 4..8 {
            let entry = DynamicEmbeddingEntry {
                inner: vec![0.0_f32; 8],
                embedding_dim: 8,
                sign: i,
            };
            map.insert(&i, entry, &String::from("first_slot"));
        }

        assert_eq!(map.len(), 12);
        assert_eq!(map.get_refresh(&3).is_none(), true);
        assert_eq!(map.get_refresh(&4).is_some(), true);

        map.insert(
            &8,
            DynamicEmbeddingEntry {
                inner: vec![0.0_f32; 8],
                embedding_dim: 8,
                sign: 8,
            },
            &String::from("first_slot"),
        );

        assert_eq!(map.len(), 12);
        assert_eq!(map.get(&5).is_none(), true);
        assert_eq!(map.get(&6).is_some(), true);
    }
}
