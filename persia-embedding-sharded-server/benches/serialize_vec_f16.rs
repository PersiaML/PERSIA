#![feature(test)]
extern crate test;

extern crate persia_embedding_sharded_server;
use persia_embedding_sharded_server::sharded_middleware_service::FeatureSumEmbeddingBatch;
// use rand::{thread_rng, Rng};
use half::f16;
use persia_speedy::{Readable, Writable};

#[derive(Readable, Writable, Debug)]
pub struct VecFeatureEmbedding {
    pub embeddings: Vec<f16>,
}

fn get_vec() -> Vec<f16> {
    let mut vec = Vec::new();
    let test_scale = 578000;
    for i in 0..test_scale {
        vec.push(f16::from_f32(i as f32 / 100.));
    }
    vec
}

#[bench]
fn allocate_zero_vec(b: &mut test::Bencher) {
    let vec = get_vec();
    b.iter(|| {
        let size = std::mem::size_of::<f16>() * vec.len();
        let _buffer: Vec<u8> = vec![0; size];
    })
}

#[bench]
fn bincode_serialize_ndarray(b: &mut test::Bencher) {
    let vec = get_vec();
    let embedding = unsafe {
        FeatureSumEmbeddingBatch {
            feature_name: "".to_string(),
            embeddings: ndarray::Array2::from_shape_vec_unchecked((1, vec.len()), vec),
        }
    };
    b.iter(|| {
        ::bincode::serialize(&embedding).unwrap();
    })
}

#[bench]
fn serialize_vec_speedy(b: &mut test::Bencher) {
    let vec = get_vec();
    let embedding = VecFeatureEmbedding { embeddings: vec };
    b.iter(|| {
        std::hint::black_box(embedding.write_to_vec().unwrap());
    })
}

#[bench]
fn bincode_serialize_vec(b: &mut test::Bencher) {
    let vec = get_vec();
    b.iter(|| {
        bincode::serialize(&vec).unwrap();
    })
}

#[bench]
fn ptr_copy(b: &mut test::Bencher) {
    let vec = get_vec();
    let vector_size = vec.len();
    b.iter(|| {
        let size = std::mem::size_of::<f16>() * vector_size;
        let mut buffer: Vec<u8> = Vec::with_capacity(size);
        buffer.extend_from_slice(unsafe {
            std::slice::from_raw_parts(vec.as_ptr() as *const u8, size)
        });
    })
}

#[bench]
fn ptr_copy_with_zero_vec_no_alloc(b: &mut test::Bencher) {
    let vec = get_vec();
    let size = std::mem::size_of::<f16>() * vec.len();
    let mut buffer: Vec<u8> = vec![0; size];
    b.iter(|| unsafe {
        std::hint::black_box(std::ptr::copy_nonoverlapping(
            vec.as_ptr() as *const u8,
            buffer.as_mut_ptr(),
            size,
        ));
    })
}
