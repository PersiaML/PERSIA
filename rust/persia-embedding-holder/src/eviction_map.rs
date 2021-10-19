use persia_libs::hashbrown::HashMap;
use std::convert::TryFrom;
use std::hash::Hash;

use crate::array_linked_list::ArrayLinkedList;

pub trait EvictionMapValue<K> {
    fn hashmap_key(&self) -> K;
}

pub struct EvictionMap<K, V>
where
    K: Hash + Eq + Clone,
    V: EvictionMapValue<K>,
{
    pub hashmap: HashMap<K, u32>,
    pub linkedlist: ArrayLinkedList<V>,
    pub capacity: usize,
}

impl<K, V> EvictionMap<K, V>
where
    K: Hash + Eq + Clone,
    V: EvictionMapValue<K>,
{
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            hashmap: HashMap::with_capacity(capacity + 1),
            linkedlist: ArrayLinkedList::with_capacity(capacity as u32 + 1),
            capacity,
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        match self.hashmap.get(&key) {
            Some(idx) => self.linkedlist[*idx as usize].as_ref(),
            None => None,
        }
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self.hashmap.get(&key) {
            Some(idx) => self.linkedlist[*idx as usize].as_mut(),
            None => None,
        }
    }

    pub fn get_refresh(&mut self, key: &K) -> Option<&V> {
        match self.hashmap.get(&key) {
            Some(idx) => {
                let idx = u32::try_from(*idx).expect("u32 array linked list overflow");
                let v = self.linkedlist.remove(idx).unwrap();
                let new_idx = self.linkedlist.push_back(v);
                let idx_ref = self.hashmap.get_mut(key).unwrap();
                *idx_ref = new_idx;
                self.linkedlist[new_idx as usize].as_ref()
            }
            None => None,
        }
    }

    pub fn get_refresh_mut(&mut self, key: &K) -> Option<&mut V> {
        match self.hashmap.get(&key) {
            Some(idx) => {
                let idx = u32::try_from(*idx).expect("u32 array linked list overflow");
                let v = self.linkedlist.remove(idx).unwrap();
                let new_idx = self.linkedlist.push_back(v);
                let idx_ref = self.hashmap.get_mut(key).unwrap();
                *idx_ref = new_idx;
                self.linkedlist[new_idx as usize].as_mut()
            }
            None => None,
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> (Option<V>, Option<V>) {
        let old = match self.hashmap.get(&key) {
            Some(idx) => self.linkedlist.remove(*idx),
            None => None,
        };

        let new_idx = self.linkedlist.push_back(value);
        self.hashmap.insert(key, new_idx);

        let evicted = if self.linkedlist.len() as usize > self.capacity {
            let evicted = self.linkedlist.pop_front();
            if let Some(evicted_v) = &evicted {
                let evicted_k = evicted_v.hashmap_key();
                self.hashmap.remove(&evicted_k);
            }
            evicted
        } else {
            None
        };

        (old, evicted)
    }

    pub fn clear(&mut self) {
        self.hashmap.clear();
        self.linkedlist.clear();
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.linkedlist.len() as usize
    }
}

#[cfg(test)]
mod eviction_map_tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::emb_entry::HashMapEmbeddingEntry;
    use persia_embedding_config::InitializationMethod;

    #[test]
    fn test_evict() {
        let mut map: EvictionMap<u64, HashMapEmbeddingEntry> = EvictionMap::with_capacity(5);

        let initialization = InitializationMethod::default();

        for i in 0..5 {
            let entry = HashMapEmbeddingEntry::new(&initialization, 8, 16, i, i);
            map.insert(i, entry);
        }

        assert_eq!(map.len(), 5);

        for i in 5..10 {
            let entry = HashMapEmbeddingEntry::new(&initialization, 8, 16, i, i);
            map.insert(i, entry);
        }

        assert_eq!(map.len(), 5);
        assert_eq!(map.get_refresh(&4).is_none(), true);
        assert_eq!(map.get_refresh(&5).is_some(), true);

        let entry = HashMapEmbeddingEntry::new(&initialization, 8, 16, 10, 10);
        map.insert(10, entry);

        assert_eq!(map.len(), 5);
        assert_eq!(map.get_refresh(&6).is_none(), true);
        assert_eq!(map.get_refresh(&5).is_some(), true);
    }
}
