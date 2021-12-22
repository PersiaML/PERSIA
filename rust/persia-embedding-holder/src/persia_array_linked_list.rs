use crate::emb_entry::{
    DynamicEmbeddingEntry, PersiaEmbeddingEntry, PersiaEmbeddingEntryMut,
    PersiaEmbeddingEntryRef,
};
use persia_libs::serde::{self, Deserialize, Serialize};
use persia_speedy::{Context, Readable, Writable};
use std::{hint, mem};

pub trait PersiaArrayLinkedList {
    fn get(&self, key: u32) -> Option<PersiaEmbeddingEntryRef>;

    fn get_mut(&mut self, key: u32) -> Option<PersiaEmbeddingEntryMut>;

    fn move_to_back(&mut self, index: u32) -> u32;

    fn remove(&mut self, index: u32) -> Option<u64>;

    fn push_back(&mut self, value: DynamicEmbeddingEntry) -> u32;

    fn pop_front(&mut self) -> Option<u64>;

    fn clear(&mut self);

    fn len(&self) -> usize;
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "self::serde")]
pub struct LinkedListNode<T> {
    next_index: u32,
    prev_index: u32,
    data: Option<T>,
}

impl<T> LinkedListNode<T> {
    fn _new(prev_index: u32, next_index: u32, data: T) -> Self {
        Self {
            next_index: next_index as _,
            prev_index: prev_index as _,
            data: Some(data),
        }
    }

    fn _front(first_index: u32, data: T) -> Self {
        Self {
            next_index: first_index as _,
            prev_index: 0,
            data: Some(data),
        }
    }

    fn back(last_index: u32, data: T) -> Self {
        Self {
            next_index: 0,
            prev_index: last_index as _,
            data: Some(data),
        }
    }

    fn deleted(free_index: u32) -> Self {
        Self {
            next_index: free_index as _,
            prev_index: 0,
            data: None,
        }
    }
}

impl<'a, C, T> Readable<'a, C> for LinkedListNode<T>
where
    C: Context,
    T: Readable<'a, C>,
{
    #[inline]
    fn read_from<R: persia_speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        let next_index: u32 = reader.read_value()?;
        let prev_index: u32 = reader.read_value()?;
        let data: Option<T> = {
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

impl<C, T> Writable<C> for LinkedListNode<T>
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "self::serde")]
pub struct ArrayLinkedList<T> {
    count: usize,
    first_index: u32,
    last_index: u32,
    free_index: u32,
    end_index: u32,
    elements: Vec<LinkedListNode<T>>,
}

impl<T> ArrayLinkedList<T> {
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

    pub fn with_capacity(capacity: usize) -> Self {
        let mut result = Self::new();
        result.elements = Vec::with_capacity(capacity as _);
        result.fill_elements(capacity as _);
        result
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

    fn insert_free_element(&mut self, element: LinkedListNode<T>) -> u32 {
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

    fn remove_index(&mut self, index: u32) -> Option<T> {
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

        data
    }

    pub fn push_back_value(&mut self, value: T) -> u32 {
        let element = LinkedListNode::back(self.last_index as _, value);

        let prev_index = self.insert_free_element(element);

        *self.next_of_prev(self.last_index as _, true) = prev_index;

        self.last_index = prev_index;
        self.count += 1;

        prev_index - 1
    }

    pub fn pop_front_value(&mut self) -> Option<T> {
        if self.first_index == 0 {
            return None;
        }
        let index = self.first_index - 1;
        let LinkedListNode {
            next_index, data, ..
        } = mem::replace(
            &mut self.elements[index as usize],
            LinkedListNode::deleted(self.free_index),
        );

        *self.prev_of_next(next_index, true) = 0;
        self.first_index = next_index;

        self.count -= 1;
        if self.free_index > 0 {
            self.elements[self.free_index as usize - 1].prev_index = index;
        }
        self.free_index = index;
        Some(data.unwrap_or_else(|| unsafe { hint::unreachable_unchecked() }))
    }
}

impl<T: PersiaEmbeddingEntry> PersiaArrayLinkedList for ArrayLinkedList<T> {
    fn get(&self, key: u32) -> Option<PersiaEmbeddingEntryRef> {
        match &self.elements[key as usize].data {
            Some(entry) => Some(PersiaEmbeddingEntryRef {
                inner: entry.get_ref(),
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
                    inner: entry.get_mut(),
                    embedding_dim,
                    sign,
                })
            }
            None => None,
        }
    }

    fn move_to_back(&mut self, index: u32) -> u32 {
        match self.remove_index(index) {
            Some(value) => self.push_back_value(value),
            None => unreachable!("move_to_back index not exsit"),
        }
    }

    fn remove(&mut self, index: u32) -> Option<u64> {
        let data = self.remove_index(index);

        match data {
            Some(entry) => Some(entry.sign()),
            None => None,
        }
    }

    fn push_back(&mut self, value: DynamicEmbeddingEntry) -> u32 {
        let entry = T::from_dynamic(value);
        self.push_back_value(entry)
    }

    fn pop_front(&mut self) -> Option<u64> {
        match self.pop_front_value() {
            Some(entry) => Some(entry.sign()),
            None => None,
        }
    }

    fn clear(&mut self) {
        self.count = 0;
        self.first_index = 0;
        self.last_index = 0;
        self.free_index = 0;
        self.end_index = 0;

        let capacity = self.elements.len();
        self.elements.clear();
        self.fill_elements(capacity as _);
    }

    fn len(&self) -> usize {
        self.count as _
    }
}
