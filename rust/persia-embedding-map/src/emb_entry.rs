use persia_common::optim::Optimizable;
use persia_embedding_config::InitializationMethod;
use persia_libs::{
    ndarray::Array1,
    ndarray_rand::rand_distr::{Gamma, Normal, Poisson, Uniform},
    ndarray_rand::RandomExt,
    rand::prelude::SmallRng,
    rand::SeedableRng,
    serde::{self, Deserialize, Serialize},
};
use persia_speedy::{Context, Readable, Writable};
use smallvec::SmallVec;
use std::sync::Arc;

#[derive(Serialize, Deserialize, Readable, Writable, Clone, Debug)]
#[serde(crate = "self::serde")]
pub struct DynamicEmbeddingEntry {
    pub inner: Vec<f32>,
    pub embedding_dim: usize,
    pub sign: u64,
    pub slot_index: usize,
}

pub struct PersiaEmbeddingEntryRef<'a> {
    pub inner: &'a [f32],
    pub embedding_dim: usize,
    pub sign: u64,
}

impl<'a> PersiaEmbeddingEntryRef<'a> {
    pub fn emb(&'a self) -> &'a [f32] {
        &self.inner[..self.embedding_dim]
    }

    pub fn opt(&'a self) -> &'a [f32] {
        &self.inner[self.embedding_dim..]
    }
}

pub struct PersiaEmbeddingEntryMut<'a> {
    pub inner: &'a mut [f32],
    pub embedding_dim: usize,
    pub sign: u64,
}

impl<'a> PersiaEmbeddingEntryMut<'a> {
    pub fn emb(&'a mut self) -> &'a mut [f32] {
        &mut self.inner[..self.embedding_dim]
    }

    pub fn opt(&'a mut self) -> &'a mut [f32] {
        &mut self.inner[self.embedding_dim..]
    }
}

pub trait PersiaEmbeddingEntry {
    fn size() -> usize;

    fn new(
        initialization_method: &InitializationMethod,
        embedding_space: usize,
        optimizer: Arc<Box<dyn Optimizable + Send + Sync>>,
        seed: u64,
        sign: u64,
    ) -> Self;

    fn from_dynamic(dynamic_entry: DynamicEmbeddingEntry) -> Self;

    fn dim(&self) -> usize;

    fn get_ref(&self) -> &[f32];

    fn get_mut(&mut self) -> &mut [f32];

    fn get_vec(&self) -> Vec<f32>;

    fn sign(&self) -> u64;

    fn len(&self) -> usize;
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(crate = "self::serde")]
pub struct ArrayEmbeddingEntry<T, const L: usize> {
    pub inner: SmallVec<[T; L]>,
    pub embedding_dim: usize,
    pub sign: u64,
}

impl<const L: usize> PersiaEmbeddingEntry for ArrayEmbeddingEntry<f32, L> {
    fn size() -> usize {
        L
    }

    fn new(
        initialization_method: &InitializationMethod,
        embedding_space: usize,
        optimizer: Arc<Box<dyn Optimizable + Send + Sync>>,
        seed: u64,
        sign: u64,
    ) -> Self {
        let emb = {
            let mut rng = SmallRng::seed_from_u64(seed);
            match initialization_method {
                InitializationMethod::BoundedUniform(x) => Array1::random_using(
                    (embedding_space,),
                    Uniform::new(x.lower, x.upper),
                    &mut rng,
                ),
                InitializationMethod::BoundedGamma(x) => Array1::random_using(
                    (embedding_space,),
                    Gamma::new(x.shape, x.scale).unwrap(),
                    &mut rng,
                ),
                InitializationMethod::BoundedPoisson(x) => Array1::random_using(
                    (embedding_space,),
                    Poisson::new(x.lambda).unwrap(),
                    &mut rng,
                ),
                InitializationMethod::BoundedNormal(x) => Array1::random_using(
                    (embedding_space,),
                    Normal::new(x.mean, x.standard_deviation).unwrap(),
                    &mut rng,
                ),
                _ => panic!(
                    "unsupported initialization method for hashmap impl: {:?}",
                    initialization_method
                ),
            }
        };

        let mut inner = emb.into_raw_vec();
        let optimizer_space = optimizer.require_space(embedding_space);
        if optimizer_space > 0 {
            inner.resize(embedding_space + optimizer_space, 0.0_f32);
        }

        optimizer.state_initialization(inner.as_mut(), optimizer_space);

        Self {
            inner: SmallVec::<[f32; L]>::from_vec(inner),
            embedding_dim: embedding_space,
            sign,
        }
    }

    fn from_dynamic(dynamic_entry: DynamicEmbeddingEntry) -> Self {
        let sign = dynamic_entry.sign;
        let embedding_dim = dynamic_entry.embedding_dim;
        Self {
            sign,
            embedding_dim,
            inner: SmallVec::<[f32; L]>::from_vec(dynamic_entry.inner),
        }
    }

    fn dim(&self) -> usize {
        self.embedding_dim
    }

    fn sign(&self) -> u64 {
        self.sign
    }

    fn get_ref(&self) -> &[f32] {
        &self.inner[..]
    }

    fn get_mut(&mut self) -> &mut [f32] {
        &mut self.inner[..]
    }

    fn get_vec(&self) -> Vec<f32> {
        self.inner.to_vec()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<C, T, const L: usize> Writable<C> for ArrayEmbeddingEntry<T, L>
where
    C: Context,
    T: Writable<C>,
{
    #[inline]
    fn write_to<W: ?Sized + persia_speedy::Writer<C>>(
        &self,
        writer: &mut W,
    ) -> Result<(), C::Error> {
        self.embedding_dim.write_to(writer)?;
        self.sign.write_to(writer)?;
        self.inner.as_slice().write_to(writer)
    }
}

impl<'a, C, T, const L: usize> Readable<'a, C> for ArrayEmbeddingEntry<T, L>
where
    C: Context,
    T: Readable<'a, C>,
{
    #[inline]
    fn read_from<R: persia_speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        let embedding_dim: usize = reader.read_value()?;
        let sign: u64 = reader.read_value()?;

        let v: Vec<T> = Readable::read_from(reader)?;
        let inner: SmallVec<[T; L]> = SmallVec::from_vec(v);

        Ok(Self {
            inner,
            embedding_dim,
            sign,
        })
    }

    #[inline]
    fn minimum_bytes_needed() -> usize {
        let mut out = 0;
        out += <usize as persia_speedy::Readable<'a, C>>::minimum_bytes_needed();
        out += <u64 as persia_speedy::Readable<'a, C>>::minimum_bytes_needed();
        out += <f32 as persia_speedy::Readable<'a, C>>::minimum_bytes_needed() * L;
        out
    }
}
