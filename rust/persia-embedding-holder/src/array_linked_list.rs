#![deny(missing_docs)]

/*!
The `ArrayLinkedList` data structure combines the benefit of an array and a linked list.
Every supported operation, which does not (re-)allocate the array, is done in *O*(1):
* inserting elements at the front and back
* popping element at the front or back
* getting the element count
* removing elements at an arbitrary index
* inserting elements at an arbitrary index
* replacing elements at an arbitrary index
It's stored like an array, but contains some additional information.
You would typically use it, where you need to be able to do multiple of the following tasks efficiently:
* accessing single elements by index
* adding and removing elements without changing order or indices
* sorting elements without changing indices or moving the content around.
# Order and indexing
You might also use it as a more convenient version of a `Vec<Option<T>>`.
When iterating over it, only the elements, which are `Some` are given to the user.
And even the checks for `Some` are optimized away.
So when it's likely, that most of the options of a large array are `None`, this might be a huge performance improvement.
Another advantage over a `LinkedList` is the cache locality.
Everything is laid out in a contiguous region of memory.
Compared to a `Vec` on the other hand, it might be bad.
The iteration does not necessarily take place in the same order.
That's mostly a problem for large arrays.
The iterator would jump back and forth in the array.
In order to understand this type, it's necessary to know about the iteration order.
There is a logical order, which is used by the iterators, or when doing anything with the first and last elements.
You can think of it as the order of a linked list, which is just packed into an array here.
And then there is indexing, which has nothing to do with the order of the linked list.
The indices just return the array elements.
## Index Example
So when adding an element to the linked array without specifying the index, you get the index, it was put to, as a result.
The results are always added to the array in order, so the indices increase, no matter if you add the indices to the front or to the back:
```
use array_linked_list::ArrayLinkedList;
let mut array = ArrayLinkedList::new();
assert_eq!(array.push_front(1), 0);
assert_eq!(array.push_back(2), 1);
assert_eq!(array.push_front(3), 2);
assert_eq!(array.push_front(4), 3);
assert_eq!(array.push_back(5), 4);
```
## Order example
When you just apped elements from the front or back, the indices even correlate to the order:
```
use array_linked_list::ArrayLinkedList;
let mut array = ArrayLinkedList::new();
array.push_front(1);
array.push_front(2);
array.push_front(3);
for (i, element) in array.iter().rev().enumerate() {
    assert_eq!(*element, array[i].unwrap());
}
```
```
use array_linked_list::ArrayLinkedList;
let mut array = ArrayLinkedList::new();
array.push_back(1);
array.push_back(2);
array.push_back(3);
for (i, element) in array.iter().enumerate() {
    assert_eq!(*element, array[i].unwrap());
}
```
## Iteration over unsorted lists
In realistic cases, you need to store the indices somewhere else, if you need them.
Alternatively, you can also use
```
use array_linked_list::ArrayLinkedList;
let mut array = ArrayLinkedList::new();
array.push_back(1);
array.push_front(2);
array.push_front(3);
array.push_back(4);
array.push_front(5);
for (index, element) in array.indexed().rev() {
    assert_eq!(*element, array[index].unwrap());
}
```
## Conclusion
Just remember, that indices and order are two different things, which don't correlate, and you should be safe.
**/

use std::{
    hint, mem,
    ops::{Index, IndexMut},
};

use persia_libs::serde::{self, Deserialize, Serialize};
use persia_speedy::{Context, Readable, Writable};

/// The `LinkedListNode` type, which is elements of `ArrayLinkedList`.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "self::serde")]
pub struct LinkedListNode<T> {
    next_index: u32,
    prev_index: u32,
    data: Option<T>,
}

impl<T> LinkedListNode<T> {
    fn new(prev_index: u32, next_index: u32, data: T) -> Self {
        Self {
            next_index: next_index as _,
            prev_index: prev_index as _,
            data: Some(data),
        }
    }

    fn front(first_index: u32, data: T) -> Self {
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
                    Ok(Some(reader.read_value()?))
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
            writer.write_value(data)?;
        } else {
            writer.write_u8(0)?;
        }

        Ok(())
    }
}

/// The `ArrayLinkedList` type, which combines the advantages of dynamic arrays and linked lists.
#[derive(Clone, Debug, Serialize, Deserialize, Readable, Writable)]
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
    /// Constructs a new, empty ArrayLinkedList<T>.
    ///
    /// The linked array will not allocate until elements are pushed onto it.
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
        self.end_index = capacity as _;
    }

    /// Constructs a new, empty `ArrayLinkedList<T>` with the specified capacity.
    ///
    /// The array will be able to hold exactly `capacity` elements without reallocating.
    // If `capacity` is 0, the vector will not allocate.
    pub fn with_capacity(capacity: u32) -> Self {
        let mut result = Self::new();
        result.elements = Vec::with_capacity(capacity as _);
        result.fill_elements(capacity);
        result
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

    /// Adds an element at the front of the array and returns its index.
    /// The indices are returned in a raising order, starting with zero.
    /// See the module description for more information.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use array_linked_list::ArrayLinkedList;
    ///
    /// let mut array = ArrayLinkedList::new();
    ///
    /// assert_eq!(array.push_front(2), 0);
    /// assert_eq!(array.front().unwrap(), &2);
    ///
    /// assert_eq!(array.push_front(1), 1);
    /// assert_eq!(array.front().unwrap(), &1);
    /// ```
    pub fn push_front(&mut self, value: T) -> u32 {
        let element = LinkedListNode::front(self.first_index as _, value);

        let next_index = self.insert_free_element(element);

        *self.prev_of_next(self.first_index as _, true) = next_index as _;

        self.first_index = next_index as _;
        self.count += 1;

        next_index - 1
    }

    /// Adds an element at the back of the array and returns its index.
    /// The indices are returned in a raising order, starting with zero.
    /// See the module description for more information.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use array_linked_list::ArrayLinkedList;
    ///
    /// let mut array = ArrayLinkedList::new();
    ///
    /// assert_eq!(array.push_back(1), 0);
    /// assert_eq!(array.push_back(3), 1);
    /// assert_eq!(3, *array.back().unwrap());
    /// ```
    pub fn push_back(&mut self, value: T) -> u32 {
        let element = LinkedListNode::back(self.last_index as _, value);

        let prev_index = self.insert_free_element(element);

        *self.next_of_prev(self.last_index as _, true) = prev_index;

        self.last_index = prev_index;
        self.count += 1;

        prev_index - 1
    }

    fn insert_between(&mut self, prev_index: u32, next_index: u32, value: T) -> u32 {
        let element = LinkedListNode::new(prev_index, next_index, value);

        let index = self.insert_free_element(element);

        *self.next_of_prev(prev_index, true) = index;
        *self.prev_of_next(next_index, true) = index;

        self.count += 1;

        index - 1
    }

    /// Inserts an element after the element at the specified index.
    /// Returns the index of the inserted element on success.
    /// If no element was found at the specified index, `None` is returned.
    ///
    /// # Panics
    /// Panics if prev_index >= capacity
    ///
    /// # Examples
    ///
    /// ```
    /// use array_linked_list::ArrayLinkedList;
    ///
    /// let mut array = ArrayLinkedList::new();
    ///
    /// let first = array.push_back(1);
    /// let second = array.push_back(2);
    /// let third = array.push_back(3);
    ///
    /// array.insert_after(second, 100);
    ///
    /// assert_eq!(array.pop_front(), Some(1));
    /// assert_eq!(array.pop_front(), Some(2));
    /// assert_eq!(array.pop_front(), Some(100));
    /// assert_eq!(array.pop_front(), Some(3));
    /// assert_eq!(array.pop_front(), None);
    /// ```
    pub fn insert_after(&mut self, prev_index: u32, value: T) -> Option<u32> {
        let LinkedListNode {
            next_index, data, ..
        } = &self.elements[prev_index as usize];

        if data.is_some() {
            let next_index = *next_index;
            Some(self.insert_between(prev_index + 1, next_index as _, value))
        } else {
            None
        }
    }

    /// Inserts an element before the element at the specified index.
    /// Returns the index of the inserted element on success.
    /// If no element was found at the specified index, `None` is returned.
    ///
    /// # Panics
    /// Panics if next_index >= capacity
    ///
    /// # Examples
    ///
    /// ```
    /// use array_linked_list::ArrayLinkedList;
    ///
    /// let mut array = ArrayLinkedList::new();
    ///
    /// let first = array.push_back(1);
    /// let second = array.push_back(2);
    /// let third = array.push_back(3);
    ///
    /// array.insert_before(second, 100);
    ///
    /// assert_eq!(array.pop_front(), Some(1));
    /// assert_eq!(array.pop_front(), Some(100));
    /// assert_eq!(array.pop_front(), Some(2));
    /// assert_eq!(array.pop_front(), Some(3));
    /// assert_eq!(array.pop_front(), None);
    /// ```
    pub fn insert_before(&mut self, next_index: u32, value: T) -> Option<u32> {
        let LinkedListNode {
            prev_index, data, ..
        } = &self.elements[next_index as usize];

        if data.is_some() {
            let prev_index = *prev_index;
            Some(self.insert_between(prev_index as _, next_index + 1, value))
        } else {
            None
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

    /// Removes the element at the given index and returns it, or `None` if it is empty.
    /// The indices of other items are not changed.
    /// Indices, which have never been used (see `capacity`), will not be available, but panic instead.
    ///
    /// Indices are not the position they appear in, when iterating over them.
    /// So you can't use enumerate to get the index to delete.
    /// But the iteration order of the elements (in both directions) is preserved.
    /// See the module description for more information.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Panics
    /// Panics if index >= capacity
    ///
    /// # Examples
    ///
    /// ```
    /// use array_linked_list::ArrayLinkedList;
    ///
    /// let mut array = ArrayLinkedList::new();
    ///
    /// let first = array.push_front(1);
    /// let second = array.push_back(2);
    /// let third = array.push_front(3);
    ///
    ///
    /// assert_eq!(array.len(), 3);
    ///
    /// assert_eq!(array.remove(second).unwrap(), 2);
    /// assert_eq!(array[second], None);
    /// assert_eq!(array.len(), 2);
    /// assert_eq!(array.remove(second), None);
    /// assert_eq!(array.len(), 2);
    ///
    /// assert_eq!(array.remove(first).unwrap(), 1);
    /// assert_eq!(array.len(), 1);
    /// assert_eq!(array.remove(third).unwrap(), 3);
    /// assert_eq!(array.len(), 0);
    /// assert!(array.is_empty());
    /// ```
    pub fn remove(&mut self, index: u32) -> Option<T> {
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

    /// Adds element at specified index at the front of the list.
    /// Useful for updating contents.
    ///
    /// It basically does the same as `remove` and `push_back`, even if the specified index is already removed.
    ///
    /// # Panics
    /// Panics if index >= capacity
    ///
    /// # Examples
    ///
    /// ```
    /// use array_linked_list::ArrayLinkedList;
    ///
    /// let mut array = ArrayLinkedList::new();
    ///
    /// array.push_front(1);
    /// let first_index = array.push_back(2);
    /// array.push_front(3);
    /// let second_index = array.push_front(4);
    /// array.push_back(5);
    ///
    /// let mut array2 = array.clone();
    /// assert_eq!(array, array2);
    ///
    /// let first_element = array.replace_front(first_index, 100);
    /// let first_element2 = array2.remove(first_index);
    /// array2.push_front(100);
    /// assert_eq!(first_element, first_element2);
    /// assert_eq!(array, array2);
    ///
    /// let second_element = array.replace_front(first_index, 0);
    /// let second_element2 = array2.remove(first_index);
    /// array2.push_back(0);
    /// assert_eq!(second_element, second_element2);
    /// assert_ne!(array, array2);
    ///
    /// assert_eq!(array.len(), 5);
    /// assert_eq!(array2.len(), 5);
    /// ```
    pub fn replace_front(&mut self, index: u32, value: T) -> Option<T> {
        let LinkedListNode {
            next_index,
            prev_index,
            data,
        } = mem::replace(
            &mut self.elements[index as usize],
            LinkedListNode::front(self.first_index, value),
        );

        let removed = data.is_some();
        self.connect_indices(prev_index, next_index, removed);

        if !removed {
            self.count += 1;
        }

        if self.first_index > 0 {
            self.elements[self.first_index as usize - 1].prev_index = index + 1;
        }

        self.first_index = index + 1;
        data
    }

    /// Adds element at specified index at the front of the list.
    /// Useful for updating contents.
    ///
    /// It basically does the same as `remove` and `push_back`, even if the specified index is already removed.
    ///
    /// # Panics
    /// Panics if index >= capacity
    ///
    /// # Examples
    ///
    /// ```
    /// use array_linked_list::ArrayLinkedList;
    ///
    /// let mut array = ArrayLinkedList::new();
    ///
    /// array.push_front(1);
    /// array.push_back(2);
    /// let middle_index = array.push_back(3);
    /// array.push_front(4);
    /// array.push_back(5);
    ///
    /// let mut array2 = array.clone();
    /// assert_eq!(array, array2);
    ///
    /// let element = array.replace_back(middle_index, 100);
    /// let element2 = array2.remove(middle_index);
    /// array2.push_back(100);
    /// assert_eq!(element, element2);
    /// assert_eq!(array, array2);
    ///
    /// assert_eq!(array.len(), 5);
    /// assert_eq!(array2.len(), 5);
    /// ```
    pub fn replace_back(&mut self, index: u32, value: T) -> Option<T> {
        let LinkedListNode {
            next_index,
            prev_index,
            data,
        } = mem::replace(
            &mut self.elements[index as usize],
            LinkedListNode::back(self.last_index, value),
        );

        let removed = data.is_some();
        self.connect_indices(prev_index, next_index, removed);

        if !removed {
            self.count += 1;
        }

        if self.last_index > 0 {
            self.elements[self.last_index as usize - 1].next_index = index + 1;
        }

        self.last_index = index + 1;
        data
    }

    /// Removes the first element from the array and returns it, or `None` if it is empty.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use array_linked_list::ArrayLinkedList;
    ///
    /// let mut array = ArrayLinkedList::new();
    /// assert_eq!(array.pop_front(), None);
    /// array.push_back(1);
    /// array.push_back(3);
    /// assert_eq!(array.pop_front(), Some(1));
    /// ```
    pub fn pop_front(&mut self) -> Option<T> {
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

    /// Removes the last element from the array and returns it, or `None` if it is empty.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use array_linked_list::ArrayLinkedList;
    ///
    /// let mut array = ArrayLinkedList::new();
    /// assert_eq!(array.pop_back(), None);
    /// array.push_back(1);
    /// array.push_back(3);
    /// assert_eq!(array.pop_back(), Some(3));
    /// ```
    pub fn pop_back(&mut self) -> Option<T> {
        if self.last_index == 0 {
            return None;
        }
        let index = self.last_index - 1;
        let LinkedListNode {
            prev_index, data, ..
        } = mem::replace(
            &mut self.elements[index as usize],
            LinkedListNode::deleted(self.free_index),
        );

        self.last_index = prev_index;
        *self.next_of_prev(prev_index, true) = 0;

        self.count -= 1;
        if self.free_index > 0 {
            self.elements[self.free_index as usize - 1].prev_index = index;
        }
        self.free_index = index;
        Some(data.unwrap_or_else(|| unsafe { hint::unreachable_unchecked() }))
    }

    /// The index of the first list element.
    /// Returns `None` if array is empty.
    pub fn front_index(&self) -> Option<u32> {
        if self.first_index > 0 {
            Some(self.first_index - 1)
        } else {
            None
        }
    }

    /// The index of the last list element.
    /// Returns `None` if array is empty.
    pub fn back_index(&self) -> Option<u32> {
        if self.last_index > 0 {
            Some(self.last_index - 1)
        } else {
            None
        }
    }

    /// The first list element.
    /// Returns `None` if array is empty.
    pub fn front(&self) -> Option<&T> {
        if self.first_index > 0 {
            Some(
                self.elements[self.first_index as usize - 1]
                    .data
                    .as_ref()
                    .unwrap_or_else(|| unsafe { hint::unreachable_unchecked() }),
            )
        } else {
            None
        }
    }

    /// The last list element.
    /// Returns `None` if array is empty.
    pub fn back(&self) -> Option<&T> {
        if self.last_index > 0 {
            Some(
                self.elements[self.last_index as usize - 1]
                    .data
                    .as_ref()
                    .unwrap_or_else(|| unsafe { hint::unreachable_unchecked() }),
            )
        } else {
            None
        }
    }

    /// The first list element as a mutable reference.
    /// Returns `None` if array is empty.
    pub fn front_mut(&mut self) -> Option<&mut T> {
        if self.first_index > 0 {
            Some(
                self.elements[self.first_index as usize - 1]
                    .data
                    .as_mut()
                    .unwrap_or_else(|| unsafe { hint::unreachable_unchecked() }),
            )
        } else {
            None
        }
    }

    /// The last list element as a mutable reference.
    /// Returns `None` if array is empty.
    pub fn back_mut(&mut self) -> Option<&mut T> {
        if self.last_index > 0 {
            Some(
                self.elements[self.last_index as usize - 1]
                    .data
                    .as_mut()
                    .unwrap_or_else(|| unsafe { hint::unreachable_unchecked() }),
            )
        } else {
            None
        }
    }

    /// Checks if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.first_index == 0 && self.last_index == 0
    }

    /// Clears the linked array, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity of the array.
    /// So all indices, which have already been used (see `capacity`), are still available.
    pub fn clear(&mut self) {
        self.count = 0;
        self.first_index = 0;
        self.last_index = 0;
        self.free_index = 0;
        self.end_index = 0;

        let capacity = self.elements.len();
        self.elements.clear();
        self.fill_elements(capacity as _);
    }

    /// Returns the number of elements in the linked array.
    pub fn len(&self) -> u32 {
        self.count as _
    }

    /// Returns the number of elements the vector can hold without reallocating.
    ///
    /// Methods, which take indices, require the specified index to be below the capacity.
    ///
    /// All the following methods require indices:
    ///
    /// * `insert_before`
    /// * `insert_after`
    /// * `remove`
    /// * `replace_front`
    /// * `replace_back`
    ///
    /// Besides that, some of the iterators are constructed using indices in the same range.
    pub fn capacity(&self) -> u32 {
        self.elements.len() as _
    }

    /// Returns a borrowing iterator over its elements.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> + DoubleEndedIterator<Item = &'a T> {
        Values(Indexed::new(self))
    }

    /// Returns a borrowing iterator over its indexed elements.
    pub fn indexed<'a>(
        &'a self,
    ) -> impl Iterator<Item = (u32, &'a T)> + DoubleEndedIterator<Item = (u32, &'a T)> {
        Indexed::new(self)
    }

    /// Returns a borrowing iterator over its indices.
    pub fn indices<'a>(&'a self) -> impl Iterator<Item = u32> + DoubleEndedIterator + 'a {
        Indices(Indexed::new(self))
    }

    /// Returns a borrowing iterator over its elements, starting after the element at the specified index.
    ///
    /// # Panics
    /// Panics if index >= capacity
    pub fn iter_after<'a>(
        &'a self,
        index: u32,
    ) -> impl Iterator<Item = &'a T> + DoubleEndedIterator {
        Values(Indexed::after(self, index))
    }

    /// Returns a borrowing iterator over its indexed elements, starting after the element at the specified index.
    ///
    /// # Panics
    /// Panics if index >= capacity
    pub fn indexed_after<'a>(
        &'a self,
        index: u32,
    ) -> impl Iterator<Item = (u32, &'a T)> + DoubleEndedIterator {
        Indexed::after(self, index)
    }

    /// Returns a borrowing iterator over its indices, starting after the element at the specified index.
    ///
    /// # Panics
    /// Panics if index >= capacity
    pub fn indices_after<'a>(
        &'a self,
        index: u32,
    ) -> impl Iterator<Item = u32> + DoubleEndedIterator + 'a {
        Indices(Indexed::after(self, index))
    }

    /// Returns a borrowing iterator over its elements, ending before the element at the specified index.
    ///
    /// # Panics
    /// Panics if index >= capacity
    pub fn iter_before<'a>(
        &'a self,
        index: u32,
    ) -> impl Iterator<Item = &'a T> + DoubleEndedIterator {
        Values(Indexed::before(self, index))
    }

    /// Returns a borrowing iterator over its indexed elements, ending before the element at the specified index.
    ///
    /// # Panics
    /// Panics if index >= capacity
    pub fn indexed_before<'a>(
        &'a self,
        index: u32,
    ) -> impl Iterator<Item = (u32, &'a T)> + DoubleEndedIterator {
        Indexed::before(self, index)
    }

    /// Returns a borrowing iterator over its indices, ending before the element at the specified index.
    ///
    /// # Panics
    /// Panics if index >= capacity
    pub fn indices_before<'a>(
        &'a self,
        index: u32,
    ) -> impl Iterator<Item = u32> + DoubleEndedIterator + 'a {
        Indices(Indexed::before(self, index))
    }

    /// Returns an owning iterator returning its indexed elements.
    pub fn into_indexed(self) -> impl Iterator<Item = (u32, T)> + DoubleEndedIterator {
        IntoIndexed(self)
    }

    /// Returns an owning iterator returning its indices.
    pub fn into_indices(self) -> impl Iterator<Item = u32> + DoubleEndedIterator {
        IntoIndices(IntoIndexed(self))
    }
}

impl<T> Index<usize> for ArrayLinkedList<T> {
    type Output = Option<T>;
    fn index(&self, index: usize) -> &Option<T> {
        &self.elements[index].data
    }
}

impl<T> IndexMut<usize> for ArrayLinkedList<T> {
    fn index_mut(&mut self, index: usize) -> &mut Option<T> {
        &mut self.elements[index].data
    }
}

struct Indexed<'a, T> {
    front_index: u32,
    back_index: u32,
    array: &'a ArrayLinkedList<T>,
}

impl<'a, T> Indexed<'a, T> {
    fn empty(array: &'a ArrayLinkedList<T>) -> Self {
        Self {
            front_index: 0,
            back_index: 0,
            array,
        }
    }

    fn new(array: &'a ArrayLinkedList<T>) -> Self {
        Self {
            front_index: array.first_index,
            back_index: array.last_index,
            array,
        }
    }

    fn after(array: &'a ArrayLinkedList<T>, prev_index: u32) -> Self {
        let element = &array.elements[prev_index as usize];
        if element.data.is_some() {
            Self {
                front_index: element.next_index,
                back_index: array.last_index,
                array,
            }
        } else {
            Self::empty(array)
        }
    }

    fn before(array: &'a ArrayLinkedList<T>, next_index: u32) -> Self {
        let element = &array.elements[next_index as usize];
        if element.data.is_some() {
            Self {
                front_index: array.first_index,
                back_index: element.prev_index,
                array,
            }
        } else {
            Self::empty(array)
        }
    }
}

struct Indices<'a, T>(Indexed<'a, T>);

/// Borrowing iterator over values of the linked array.
pub struct Values<'a, T>(Indexed<'a, T>);

impl<'a, T> Iterator for Indexed<'a, T> {
    type Item = (u32, &'a T);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front_index > 0 {
            let index = self.front_index - 1;
            let element = &self.array.elements[index as usize];
            if self.front_index == self.back_index {
                self.front_index = 0;
                self.back_index = 0;
            } else {
                self.front_index = element.next_index;
            }
            Some((
                index,
                element
                    .data
                    .as_ref()
                    .unwrap_or_else(|| unsafe { hint::unreachable_unchecked() }),
            ))
        } else {
            None
        }
    }
}

impl<'a, T> DoubleEndedIterator for Indexed<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.back_index > 0 {
            let index = self.back_index - 1;
            let element = &self.array.elements[index as usize];
            if self.front_index == self.back_index {
                self.front_index = 0;
                self.back_index = 0;
            } else {
                self.back_index = element.prev_index;
            }
            Some((
                index,
                element
                    .data
                    .as_ref()
                    .unwrap_or_else(|| unsafe { hint::unreachable_unchecked() }),
            ))
        } else {
            None
        }
    }
}

impl<'a, T> Iterator for Indices<'a, T> {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((index, _)) = self.0.next() {
            Some(index)
        } else {
            None
        }
    }
}

impl<'a, T> DoubleEndedIterator for Indices<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some((index, _)) = self.0.next_back() {
            Some(index)
        } else {
            None
        }
    }
}

impl<'a, T> Iterator for Values<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((_, value)) = self.0.next() {
            Some(value)
        } else {
            None
        }
    }
}

impl<'a, T> DoubleEndedIterator for Values<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some((_, value)) = self.0.next_back() {
            Some(value)
        } else {
            None
        }
    }
}

impl<'a, T> IntoIterator for &'a ArrayLinkedList<T> {
    type Item = &'a T;
    type IntoIter = Values<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Values(Indexed::new(self))
    }
}

struct IntoIndexed<T>(ArrayLinkedList<T>);
struct IntoIndices<T>(IntoIndexed<T>);

/// Owning iterator over values of the linked array.
pub struct IntoValues<T>(IntoIndexed<T>);

impl<T> Iterator for IntoIndexed<T> {
    type Item = (u32, T);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.first_index > 0 {
            let index = self.0.first_index - 1;
            let element = &mut self.0.elements[index as usize];
            if self.0.first_index == self.0.last_index {
                self.0.first_index = 0;
                self.0.last_index = 0;
            } else {
                self.0.first_index = element.next_index;
            }
            Some((
                index,
                element
                    .data
                    .take()
                    .unwrap_or_else(|| unsafe { hint::unreachable_unchecked() }),
            ))
        } else {
            None
        }
    }
}

impl<T> DoubleEndedIterator for IntoIndexed<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.0.first_index > 0 {
            let index = self.0.first_index - 1;
            let element = &mut self.0.elements[index as usize];
            if self.0.first_index == self.0.last_index {
                self.0.first_index = 0;
                self.0.last_index = 0;
            } else {
                self.0.first_index = element.next_index;
            }
            Some((
                index,
                element
                    .data
                    .take()
                    .unwrap_or_else(|| unsafe { hint::unreachable_unchecked() }),
            ))
        } else {
            None
        }
    }
}

impl<T> Iterator for IntoIndices<T> {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((index, _)) = self.0.next() {
            Some(index)
        } else {
            None
        }
    }
}

impl<T> DoubleEndedIterator for IntoIndices<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some((index, _)) = self.0.next_back() {
            Some(index)
        } else {
            None
        }
    }
}

impl<T> Iterator for IntoValues<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((_, value)) = self.0.next() {
            Some(value)
        } else {
            None
        }
    }
}

impl<T> DoubleEndedIterator for IntoValues<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some((_, value)) = self.0.next_back() {
            Some(value)
        } else {
            None
        }
    }
}

impl<T> IntoIterator for ArrayLinkedList<T> {
    type Item = T;
    type IntoIter = IntoValues<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoValues(IntoIndexed(self))
    }
}
