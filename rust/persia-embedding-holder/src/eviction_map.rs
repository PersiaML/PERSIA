use persia_libs::hashbrown::HashMap;
use std::convert::TryFrom;
use std::hash::Hash;

use crate::array_linked_list::ArrayLinkedList;
// use crate::emb_entry::PersiaEmbeddingEntry;

// pub trait PersiaEvictionMap<K, V>
// where
//     K: Hash + Eq + Clone,
//     V: PersiaEmbeddingEntry,
// {
//     fn with_capacity(capacity: usize) -> Self;

//     fn get(&self, key: &K) -> Option<&V>;
// }

// pub struct PersiaEmbeddingRef<'a> {
//     pub inner: &'a [f32],
//     pub embedding_dim: usize,
//     pub sign: u64,
// }

// impl PersiaEmbeddingEntry for PersiaEmbeddingRef {

// }

// pub struct PersiaEmbeddingMut<'a> {
//     pub inner: &'a mut [f32],
//     pub embedding_dim: usize,
//     pub sign: u64,
// }

// impl PersiaEmbeddingEntry for PersiaEmbeddingMut {

// }

// pub struct PersiaEvictionMaps {
//     pub inner: HashMap<String, Box<dyn PersiaEvictionMap<u64, PersiaEmbeddingEntry>>>
// }

// pub trait PersiaEvictionMap {
//     fn with_capacity(capacity: usize) -> Self where Self: Sized;

//     fn get(&self, key: u64) -> Option<PersiaEmbeddingRef>;

//     fn get_mut(&mut self, key: &u64) -> Option<PersiaEmbeddingMut>;

//     fn get_refresh(&mut self, key: &u64) -> Option<PersiaEmbeddingRef>;

//     fn get_refresh_mut(&mut self, key: &u64) -> Option<PersiaEmbeddingMut>;

//     fn insert(&mut self, key: u64, value: Vec<f32>) -> (PersiaEmbeddingRef, PersiaEmbeddingRef);  // not zero copy

//     fn clear(&mut self);

//     fn capacity(&self) -> usize;

//     fn len(&self) -> usize;
// }

// pub struct LruEvictionMap<K, V>
// where
//     K: Hash + Eq + Clone,
//     V: PersiaEmbeddingEntry,
// {
//     pub hashmap: HashMap<K, u32>,
//     pub linkedlist: ArrayLinkedList<V>,
//     pub capacity: usize,
// }

// impl<K, V> PersiaEvictionMap for LruEvictionMap<K, V>
// where
//     K: Hash + Eq + Clone,
//     V: PersiaEmbeddingEntry,
// {
//     fn with_capacity(capacity: usize) -> Self
//     where Self: Sized {
//         match V::type_size() {
//             1 => {todo!()},
//             2 => {todo!()},
//             _ => {todo!()},
//             // gen arms by TokenStream
//         }
//     }

//     fn get(&self, key: u64) -> Option<PersiaEmbeddingRef> {
//         match self.hashmap.get(&key) {
//             Some(idx) => {
//                 match self.linkedlist[*idx as usize].as_ref() {
//                     Some(entry) => PersiaEmbeddingRef {
//                         inner: entry.inner(),
//                         embedding_dim: entry.dim(),
//                         sign: entry.sign(),
//                     },
//                     None => None,
//                 }
//             },
//             None => None,
//         }
//     }
// }

// trait ArrayLinkedList<T> {
//     fn get(k: u64) -> &[f32];
// }

// pub struct FeatureEvictionMap
// {
//     pub hashmap: HashMap<u64, (u32, u32)>,
//     pub linkedlists: Vec<Box<dyn ArrayLinkedList>,
//     pub capacity: Vec<usize>,
// }

// enum ArrayLinkedList {
//     Dim1(ArrayLinkedList<1>),
//     Dim2(ArrayLinkedList<2>),
// }

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
