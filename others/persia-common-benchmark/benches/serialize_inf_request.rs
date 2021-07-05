#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use criterion::*;
use criterion_macro::criterion;

use persia_speedy::{Readable, Writable};
use serde::{Deserialize, Serialize};
use std::convert::TryInto;
use std::str::FromStr;

/// current inference interface's request structure
/// baseline benchmark
#[derive(Deserialize, Serialize, Debug, Readable, Writable)]
struct BatchEmbeddingsRequest {
    model_name: String,
    version_name: String,
    embedding_lookups: Vec<BatchEmbedding>,
}

#[derive(Deserialize, Serialize, Debug, Readable, Writable)]
struct BatchEmbedding {
    embed_name: String,
    batch_indices: Vec<Vec<u64>>,
}

fn create_request() -> BatchEmbeddingsRequest {
    let batch_size = 512;
    let indices_num = 5;
    let feature_nums = 35;

    let mut embedding_lookups = Vec::new();
    for feature_idx in 0..feature_nums {
        let mut be = BatchEmbedding {
            embed_name: feature_idx.to_string(),
            batch_indices: Vec::new(),
        };
        for _ in 0..batch_size {
            let mut indices = Vec::new();
            for idx in 0..indices_num {
                indices.push(idx);
            }
            be.batch_indices.push(indices)
        }
        embedding_lookups.push(be);
    }
    BatchEmbeddingsRequest {
        model_name: String::from(""),
        version_name: String::from(""),
        embedding_lookups: embedding_lookups,
    }
}

/// server side expand to origin probuf datatype
/// to avoid code modify
#[derive(Deserialize, Serialize, Debug, Readable, Writable)]
struct CompactBatchEmbeddingsRequest {
    model_name: String,
    version_name: String,
    embedding_lookups: Vec<CompactBatchEmbedding>,
    indices: Vec<u64>,
}

#[derive(Deserialize, Serialize, Debug, Readable, Writable)]
struct CompactBatchEmbedding {
    embed_name: String,
    batch_indices_offset: Vec<u64>, // len equal to batch size
}

fn create_compact_request() -> CompactBatchEmbeddingsRequest {
    let batch_size = 512;
    let indices_num = 5;
    let feature_nums = 35;

    let mut embedding_lookups = Vec::new();
    let mut indices = Vec::new();
    let mut global_indices_idx = 0;
    for feature_idx in 0..feature_nums {
        let mut be = CompactBatchEmbedding {
            embed_name: feature_idx.to_string(),
            batch_indices_offset: Vec::new(),
        };
        for _ in 0..batch_size {
            for idx in 0..indices_num {
                indices.push(idx);
                global_indices_idx += 1;
            }
            be.batch_indices_offset.push(global_indices_idx);
        }
        embedding_lookups.push(be);
    }
    CompactBatchEmbeddingsRequest {
        model_name: String::from(""),
        version_name: String::from(""),
        embedding_lookups: embedding_lookups,
        indices: indices,
    }
}

#[derive(Debug, Readable, Writable)]
struct BatchEmbeddingsRequestSmallVec {
    model_name: tinystr::TinyStr16,
    version_name: tinystr::TinyStr16,
    embedding_lookups: Vec<BatchEmbeddingSmallVec>,
}

#[derive(Debug, Readable, Writable)]
struct BatchEmbeddingSmallVec {
    embed_name: tinystr::TinyStr16,
    batch_indices: smallvec::SmallVec<[smallvec::SmallVec<[u64; 8]>; 512]>,
}

fn create_request_smallvec() -> BatchEmbeddingsRequestSmallVec {
    let batch_size = 512;
    let indices_num = 5;
    let feature_nums = 35;

    let mut embedding_lookups = Vec::new();
    for feature_idx in 0..feature_nums {
        let mut be = BatchEmbeddingSmallVec {
            embed_name: tinystr::TinyStr16::from_str(feature_idx.to_string().as_ref()).unwrap(),
            batch_indices: smallvec::SmallVec::new(),
        };
        for _ in 0..batch_size {
            let mut indices = smallvec::SmallVec::new();
            for idx in 0..indices_num {
                indices.push(idx);
            }
            be.batch_indices.push(indices)
        }
        embedding_lookups.push(be);
    }
    BatchEmbeddingsRequestSmallVec {
        model_name: tinystr::tinystr16!("ha"),
        version_name: tinystr::tinystr16!("ha"),
        embedding_lookups,
    }
}

#[criterion]
fn bench_create_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_request");
    group.bench_function("create_request", |b| {
        b.iter(|| {
            create_request();
        })
    });
    group.bench_function("create_compact_request", |b| {
        b.iter(|| {
            create_compact_request();
        })
    });
    group.bench_function("create_smallvec_request", |b| {
        b.iter(|| {
            create_request_smallvec();
        })
    });
    group.finish();
}

#[criterion]
fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    group.bench_function("normal_bincode", |b| {
        let request = create_request();
        let bytes = bincode::serialize(&request).unwrap();
        b.iter(|| {
            bincode::deserialize::<BatchEmbeddingsRequest>(&bytes).unwrap();
        })
    });
    group.bench_function("normal_speedy", |b| {
        let request = create_request();
        let bytes = request.write_to_vec().unwrap();
        b.iter(|| {
            BatchEmbeddingsRequest::read_from_buffer(bytes.as_slice()).unwrap();
        })
    });
    group.bench_function("smallvec_speedy", |b| {
        let request = create_request_smallvec();
        let bytes = request.write_to_vec().unwrap();
        b.iter(|| {
            BatchEmbeddingsRequestSmallVec::read_from_buffer(bytes.as_slice()).unwrap();
        })
    });
    group.bench_function("compact_bincode", |b| {
        let request = create_compact_request();
        let bytes = bincode::serialize(&request).unwrap();
        b.iter(|| {
            bincode::deserialize::<CompactBatchEmbeddingsRequest>(&bytes).unwrap();
        })
    });
    group.bench_function("compact_speedy", |b| {
        let request = create_compact_request();
        let bytes = request.write_to_vec().unwrap();
        b.iter(|| {
            CompactBatchEmbeddingsRequest::read_from_buffer(bytes.as_slice()).unwrap();
        })
    });
    group.bench_function("compact_tear_indices_bincode", |b| {
        let (meta, indices) = create_compact_request_with_indices();
        let mut bytes = serialize_tear_indices(&meta, &indices);
        b.iter(|| {
            deserialize_tear_indices(&mut bytes);
        })
    });
    group.finish();
}

#[criterion]
fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    group.bench_function("normal_bincode", |b| {
        let request = create_request();
        b.iter(|| {
            bincode::serialize(&request).unwrap();
        })
    });
    group.bench_function("normal_speedy", |b| {
        let request = create_request();
        b.iter(|| {
            request.write_to_vec().unwrap();
        })
    });
    group.bench_function("smallvec_speedy", |b| {
        let request = create_request_smallvec();
        b.iter(|| {
            request.write_to_vec().unwrap();
        })
    });
    group.bench_function("compact_bincode", |b| {
        let request = create_compact_request();
        b.iter(|| {
            bincode::serialize(&request).unwrap();
        })
    });
    group.bench_function("compact_speedy", |b| {
        let request = create_compact_request();
        b.iter(|| {
            request.write_to_vec().unwrap();
        })
    });
    group.bench_function("compact_tear_indices_bincode", |b| {
        let (meta, indices) = create_compact_request_with_indices();
        b.iter(|| {
            serialize_tear_indices(&meta, &indices);
        })
    });
    group.finish();
}

#[derive(Deserialize, Serialize, Debug)]
struct CompactBatchEmbeddingsRequestWithoutIndices {
    model_name: String,
    version_name: String,
    embedding_lookups: Vec<CompactBatchEmbedding>,
}

fn create_compact_request_with_indices() -> (CompactBatchEmbeddingsRequestWithoutIndices, Vec<u64>)
{
    let batch_size = 512;
    let indices_num = 5;
    let feature_nums = 35;

    let mut embedding_lookups = Vec::new();
    let mut indices = Vec::new();
    let mut global_indices_idx = 0;
    for feature_idx in 0..feature_nums {
        let mut be = CompactBatchEmbedding {
            embed_name: feature_idx.to_string(),
            batch_indices_offset: Vec::new(),
        };
        for _ in 0..batch_size {
            for idx in 0..indices_num {
                indices.push(idx);
                global_indices_idx += 1;
            }
            be.batch_indices_offset.push(global_indices_idx);
        }
        embedding_lookups.push(be);
    }
    (
        CompactBatchEmbeddingsRequestWithoutIndices {
            model_name: String::from(""),
            version_name: String::from(""),
            embedding_lookups: embedding_lookups,
        },
        indices,
    )
}

fn serialize_tear_indices(
    meta: &CompactBatchEmbeddingsRequestWithoutIndices,
    indices: &Vec<u64>,
) -> Vec<u8> {
    let meta_buffer_size = bincode::serialized_size(&meta).unwrap() as usize;
    let header_size = std::mem::size_of::<u64>();
    let indices_buffer_size = std::mem::size_of::<u64>() * indices.len();
    let buffer_size = meta_buffer_size + header_size + indices_buffer_size;
    let mut buffer = Vec::with_capacity(buffer_size);
    unsafe {
        buffer.set_len(buffer_size);
        buffer[..header_size].clone_from_slice(&(meta_buffer_size as u64).to_be_bytes());
        buffer[header_size + meta_buffer_size..].clone_from_slice(std::slice::from_raw_parts(
            indices.as_ptr() as *const u8,
            indices_buffer_size,
        ));
    }
    bincode::serialize_into(
        &mut buffer[header_size..meta_buffer_size + header_size],
        &meta,
    )
    .unwrap();
    buffer
}

fn deserialize_tear_indices(
    bytes: &mut [u8],
) -> (CompactBatchEmbeddingsRequestWithoutIndices, &[u64]) {
    let header_size = std::mem::size_of::<u64>();
    let meta_buffer_size = u64::from_be_bytes(bytes[..header_size].try_into().unwrap()) as usize;
    let meta = bincode::deserialize::<CompactBatchEmbeddingsRequestWithoutIndices>(
        &bytes[header_size..header_size + meta_buffer_size],
    )
    .unwrap();
    // meta
    let ptr = bytes[meta_buffer_size + header_size..].as_ptr() as *const u64;
    let slice = unsafe {
        std::slice::from_raw_parts(
            ptr,
            (bytes.len() - meta_buffer_size - header_size) / std::mem::size_of::<u64>(),
        )
    };
    (meta, slice)
}
