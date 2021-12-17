use crate::emb_entry::{ArrayEmbeddingEntry, PersiaEmbeddingEntry};
use persia_libs::half::f16;
use persia_libs::serde::{self, Deserialize, Serialize};
use persia_speedy::{Context, Readable, Writable};
use std::mem;

pub enum PersiaEmbeddingEntryWithType {
    F32(Box<dyn PersiaEmbeddingEntry<f32>>),
    F16(Box<dyn PersiaEmbeddingEntry<f16>>),
}

pub enum PersiaEmbeddingRef<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
}

pub enum PersiaEmbeddingMut<'a> {
    F32(&'a mut [f32]),
    F16(&'a mut [f16]),
}

pub struct PersiaEmbeddingEntryRef<'a> {
    pub inner: PersiaEmbeddingRef<'a>,
    pub embedding_dim: usize,
    pub sign: u64,
}

pub struct PersiaEmbeddingEntryMut<'a> {
    pub inner: PersiaEmbeddingMut<'a>,
    pub embedding_dim: usize,
    pub sign: u64,
}

pub trait PersiaArrayLinkedList {
    fn with_capacity(capacity: usize) -> Self;

    fn get(&self, key: u32) -> Option<PersiaEmbeddingEntryRef>;

    fn get_mut(&mut self, key: u32) -> Option<PersiaEmbeddingEntryMut>;

    fn remove(&mut self, index: u32) -> Option<PersiaEmbeddingEntryWithType>;

    fn push_back(&mut self, value: PersiaEmbeddingEntryWithType) -> u32;

    fn pop_front(&mut self) -> Option<PersiaEmbeddingEntryWithType>;

    fn clear(&mut self);

    fn len(&self) -> usize;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "self::serde")]
pub struct LinkedListNode<T, const L: usize> {
    next_index: u32,
    prev_index: u32,
    data: Option<ArrayEmbeddingEntry<T, L>>,
}

impl<T, const L: usize> LinkedListNode<T, L> {
    fn new(prev_index: u32, next_index: u32, data: ArrayEmbeddingEntry<T, L>) -> Self {
        Self {
            next_index: next_index,
            prev_index: prev_index,
            data: Some(data),
        }
    }

    fn front(first_index: u32, data: ArrayEmbeddingEntry<T, L>) -> Self {
        Self {
            next_index: first_index,
            prev_index: 0,
            data: Some(data),
        }
    }

    fn back(last_index: u32, data: ArrayEmbeddingEntry<T, L>) -> Self {
        Self {
            next_index: 0,
            prev_index: last_index,
            data: Some(data),
        }
    }

    fn deleted(free_index: u32) -> Self {
        Self {
            next_index: free_index,
            prev_index: 0,
            data: None,
        }
    }
}

impl<'a, C, T, const L: usize> Readable<'a, C> for LinkedListNode<T, L>
where
    C: Context,
    T: Readable<'a, C>,
{
    #[inline]
    fn read_from<R: persia_speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        let next_index: u32 = reader.read_value()?;
        let prev_index: u32 = reader.read_value()?;
        let data: Option<ArrayEmbeddingEntry<T, L>> = {
            reader.read_u8().and_then(|_flag_| {
                if _flag_ != 0 {
                    Ok(Some(Readable::read_from(reader)?))
                } else {
                    Ok(None)
                }
            })
        }?;

        Ok(Self {
            next_index,
            prev_index,
            data,
        })
    }

    #[inline]
    fn minimum_bytes_needed() -> usize {
        {
            let mut out = 0;
            out += <u32 as persia_speedy::Readable<'a, C>>::minimum_bytes_needed();
            out += <u32 as persia_speedy::Readable<'a, C>>::minimum_bytes_needed();
            out += 1;
            out
        }
    }
}

impl<C, T, const L: usize> Writable<C> for LinkedListNode<T, L>
where
    C: Context,
    T: Writable<C>,
{
    #[inline]
    fn write_to<W: ?Sized + persia_speedy::Writer<C>>(
        &self,
        writer: &mut W,
    ) -> Result<(), C::Error> {
        let next_index = &self.next_index;
        let prev_index = &self.prev_index;
        let data = &self.data;

        writer.write_value(next_index)?;
        writer.write_value(prev_index)?;

        if let Some(ref data) = data {
            writer.write_u8(1)?;
            data.write_to(writer)?;
        } else {
            writer.write_u8(0)?;
        }

        Ok(())
    }
}

/// The `ArrayLinkedList` type, which combines the advantages of dynamic arrays and linked lists.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "self::serde")]
pub struct ArrayLinkedList<T, const L: usize> {
    count: usize,
    first_index: u32,
    last_index: u32,
    free_index: u32,
    end_index: u32,
    elements: Vec<LinkedListNode<T, L>>,
}

impl<T, const L: usize> ArrayLinkedList<T, L> {
    pub fn new() -> Self {
        Self {
            count: 0,
            first_index: 0,
            last_index: 0,
            free_index: 0,
            end_index: 0,
            elements: Vec::new(),
        }
    }

    #[inline]
    fn fill_elements(&mut self, capacity: u32) {
        if capacity == 0 {
            return;
        }
        for i in 1..capacity {
            self.elements.push(LinkedListNode::deleted(i + 1))
        }
        self.elements.push(LinkedListNode::deleted(0));

        self.free_index = 1;
        self.end_index = capacity;
    }

    fn insert_free_element(&mut self, element: LinkedListNode<T, L>) -> u32 {
        if self.free_index == 0 {
            self.elements.push(element);
            self.elements.len() as _
        } else {
            let free_index = self.free_index;
            let recycle_element = &mut self.elements[free_index as usize - 1];
            self.free_index = recycle_element.next_index;
            *recycle_element = element;
            free_index
        }
    }

    #[inline]
    fn prev_of_next(&mut self, index: u32, active: bool) -> &mut u32 {
        if index > 0 {
            &mut self.elements[index as usize - 1].prev_index
        } else if active {
            &mut self.last_index
        } else {
            &mut self.end_index
        }
    }

    #[inline]
    fn next_of_prev(&mut self, index: u32, active: bool) -> &mut u32 {
        if index > 0 {
            &mut self.elements[index as usize - 1].next_index
        } else if active {
            &mut self.first_index
        } else {
            &mut self.free_index
        }
    }

    fn connect_indices(&mut self, prev_index: u32, next_index: u32, active: bool) {
        *self.prev_of_next(next_index, active) = prev_index;
        *self.next_of_prev(prev_index, active) = next_index;
    }
}

impl<const L: usize> PersiaArrayLinkedList for ArrayLinkedList<f32, L> {
    fn with_capacity(capacity: usize) -> Self {
        let mut result = Self::new();
        result.elements = Vec::with_capacity(capacity as _);
        result.fill_elements(capacity as _);
        result
    }

    fn get(&self, key: u32) -> Option<PersiaEmbeddingEntryRef> {
        match &self.elements[key as usize].data {
            Some(entry) => Some(PersiaEmbeddingEntryRef {
                inner: PersiaEmbeddingRef::F32(entry.inner()),
                embedding_dim: entry.dim(),
                sign: entry.sign(),
            }),
            None => None,
        }
    }

    fn get_mut(&mut self, key: u32) -> Option<PersiaEmbeddingEntryMut> {
        match &mut self.elements[key as usize].data {
            Some(entry) => {
                let embedding_dim = entry.dim();
                let sign = entry.sign();
                Some(PersiaEmbeddingEntryMut {
                    inner: PersiaEmbeddingMut::F32(entry.inner_mut()),
                    embedding_dim,
                    sign,
                })
            }
            None => None,
        }
    }

    fn remove(&mut self, index: u32) -> Option<PersiaEmbeddingEntryWithType> {
        let LinkedListNode {
            next_index,
            prev_index,
            data,
        } = mem::replace(
            &mut self.elements[index as usize],
            LinkedListNode::deleted(self.free_index),
        );

        let removed = data.is_some();
        self.connect_indices(prev_index, next_index, removed);

        if removed {
            self.count -= 1;
        }

        if self.free_index > 0 {
            self.elements[self.free_index as usize - 1].prev_index = index + 1;
        }
        self.free_index = index + 1;

        match data {
            Some(entry) => Some(PersiaEmbeddingEntryWithType::F32(Box::new(entry))),
            None => None,
        }
    }

    fn push_back(&mut self, value: PersiaEmbeddingEntryWithType) -> u32 {
        match value {
            PersiaEmbeddingEntryWithType::F32(entry) => {
                let value = *entry;
                let element = LinkedListNode::back(self.last_index as _, value);

                let prev_index = self.insert_free_element(element);

                *self.next_of_prev(self.last_index as _, true) = prev_index;

                self.last_index = prev_index;
                self.count += 1;

                prev_index - 1
            }
            PersiaEmbeddingEntryWithType::F16(_) => unreachable!(),
        }
    }

    fn pop_front(&mut self) -> PersiaEmbeddingEntryWithType {
        todo!();
    }

    fn clear(&mut self) {
        todo!();
    }

    fn len(&self) -> usize {
        todo!();
    }
}
