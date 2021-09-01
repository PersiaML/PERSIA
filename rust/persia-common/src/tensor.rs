use persia_libs::{
    itertools::Itertools,
    ndarray::Array2,
    ndarray::ShapeError,
    serde::{self, Deserialize, Serialize},
};

use persia_speedy::{Readable, Writable};

#[derive(Debug)]
pub enum TensorDtype {
    F16,
    F32,
}

#[derive(Readable, Writable, Debug)]
pub enum BaseTensor {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl BaseTensor {
    pub fn type_size(&self) -> usize {
        match &self {
            BaseTensor::F32(_) => std::mem::size_of::<f32>(),
            BaseTensor::F64(_) => std::mem::size_of::<f64>(),
            BaseTensor::I32(_) => std::mem::size_of::<i32>(),
            BaseTensor::I64(_) => std::mem::size_of::<i64>(),
        }
    }
}

#[derive(Readable, Writable, Debug)]
pub struct DenseTensor {
    pub data: BaseTensor,
    pub shape: Vec<usize>,
}

#[derive(Readable, Writable, Debug)]
pub struct SparseTensor {
    pub data: BaseTensor,
    pub offset: Vec<u64>,
}
#[derive(Readable, Writable, Debug)]
pub enum Tensor {
    Dense(DenseTensor),
    Sparse(SparseTensor),
}

#[derive(Default, Serialize, Deserialize, Readable, Writable, Debug, Clone)]
#[serde(crate = "self::serde")]
pub struct PersiaDenseTensor<T> {
    pub name: String,
    pub dim: usize,
    pub content: Vec<T>,
}

impl<T> PersiaDenseTensor<T> {
    pub fn add_sample(&mut self, mut sample: Vec<T>) {
        assert_eq!(sample.len(), self.dim);
        self.content.append(&mut sample);
    }

    pub fn empty_like(&self) -> Self {
        Self {
            name: self.name.clone(),
            dim: self.dim,
            content: vec![],
        }
    }

    pub fn get(&self, sample_idx: usize) -> &[T] {
        &self.content[sample_idx * self.dim..(sample_idx + 1) * self.dim]
    }
}

impl<T> std::convert::TryInto<Array2<T>> for PersiaDenseTensor<T> {
    type Error = ShapeError;

    fn try_into(self) -> Result<Array2<T>, Self::Error> {
        Array2::<T>::from_shape_vec((self.content.len() / self.dim, self.dim), self.content)
    }
}

impl<T> IntoIterator for PersiaDenseTensor<T> {
    type Item = Vec<T>;
    type IntoIter = PersiaDenseTensorSampleIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        assert_eq!(self.content.len() % self.dim, 0);
        let mut chunks = self
            .content
            .into_iter()
            .chunks(self.dim)
            .into_iter()
            .map(|chunk| chunk.collect_vec())
            .collect_vec();
        chunks.reverse();
        PersiaDenseTensorSampleIterator { chunks }
    }
}

pub struct PersiaDenseTensorSampleIterator<T> {
    chunks: Vec<Vec<T>>,
}

impl<T> Iterator for PersiaDenseTensorSampleIterator<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.pop()
    }
}
