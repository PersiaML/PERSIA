use super::cuda_event_pool::{CudaEventPtr, CUDA_EVENT_POOL};
use super::cuda_memory_pool::CUDA_DEVICE_MEMORY_POOL;
use super::cuda_stream_pool::CUDA_STREAM_POOL;
use super::pinned_memory_pool::PINNED_MEMORY_POOL;
use super::{AsyncEmbeddingOnCuda, AsyncRawEmbeddingOnCuda, AsyncTensorOnCuda};

use cuda_runtime_sys as cuda;
use persia_libs::{anyhow::Result, tracing};

use persia_common::{
    tensor::{BaseTensor, DenseTensor, PersiaDenseTensor, TensorDtype},
    FeatureEmbeddingBatch,
};

pub(crate) fn cuda_d2h(
    num_bytes: usize,
    data_ptr: *mut std::os::raw::c_void,
    host_ptr: *mut std::os::raw::c_void,
) -> Result<CudaEventPtr> {
    unsafe {
        let stream = CUDA_STREAM_POOL.allocate(0);
        let result = cuda::cudaMemcpyAsync(
            host_ptr,
            data_ptr,
            num_bytes,
            cuda::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            stream.inner,
        );
        assert_eq!(result, cuda::cudaError::cudaSuccess);
        let event = CUDA_EVENT_POOL.allocate(0);
        event.record(stream);
        Ok(event)
    }
}

// TODO: deprecated
pub(crate) fn cuda_tensor_h2d<T>(mut tensor: PersiaDenseTensor<T>) -> Result<AsyncTensorOnCuda> {
    unsafe {
        let stream = CUDA_STREAM_POOL.allocate(0);
        let byte_count = tensor.content.len() * std::mem::size_of::<T>();
        let host_ptr = tensor.content.as_mut_ptr() as *mut std::os::raw::c_void;
        let pinned_host_ptr = PINNED_MEMORY_POOL.allocate(byte_count);
        std::ptr::copy_nonoverlapping(host_ptr, pinned_host_ptr.inner, byte_count);
        let data_ptr = CUDA_DEVICE_MEMORY_POOL.allocate(byte_count);
        let result = cuda::cudaMemcpyAsync(
            data_ptr.inner,
            pinned_host_ptr.inner,
            byte_count,
            cuda::cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream.inner,
        );
        assert_eq!(
            result,
            cuda::cudaError::cudaSuccess,
            "data_ptr {:?}, pinned_host_ptr: {:?}, byte_count: {:?}, stream: {:?}",
            data_ptr,
            pinned_host_ptr,
            byte_count,
            stream
        );
        let event = CUDA_EVENT_POOL.allocate(0);
        event.record(stream);
        Ok(AsyncTensorOnCuda {
            name: tensor.name,
            ptr: data_ptr,
            shape: [tensor.content.len() / tensor.dim, tensor.dim],
            event,
            host_ptr: pinned_host_ptr,
            dtype: TensorDtype::F16,
        })
    }
}

pub(crate) fn cuda_dense_tensor_h2d(mut tensor: DenseTensor) -> Result<AsyncTensorOnCuda> {
    unsafe {
        let stream = CUDA_STREAM_POOL.allocate(0);
        let byte_count = tensor.shape.iter().product::<usize>() * tensor.data.type_size();

        tracing::debug!(
            "tensor shape is: {:?}, bytes: {:?}",
            &tensor.shape,
            &byte_count
        );

        let host_ptr = match &tensor.data {
            BaseTensor::F32(val) => val.as_ptr() as *const std::os::raw::c_void,
            BaseTensor::F64(val) => val.as_ptr() as *const std::os::raw::c_void,
            BaseTensor::I32(val) => val.as_ptr() as *const std::os::raw::c_void,
            BaseTensor::I64(val) => val.as_ptr() as *const std::os::raw::c_void,
        };
        let pinned_host_ptr = PINNED_MEMORY_POOL.allocate(byte_count);
        std::ptr::copy_nonoverlapping(host_ptr, pinned_host_ptr.inner, byte_count);

        let data_ptr = CUDA_DEVICE_MEMORY_POOL.allocate(byte_count);
        let result = cuda::cudaMemcpyAsync(
            data_ptr.inner,
            pinned_host_ptr.inner,
            byte_count,
            cuda::cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream.inner,
        );
        assert_eq!(
            result,
            cuda::cudaError::cudaSuccess,
            "data_ptr {:?}, pinned_host_ptr: {:?}, byte_count: {:?}, stream: {:?}",
            data_ptr,
            pinned_host_ptr,
            byte_count,
            stream
        );
        let event = CUDA_EVENT_POOL.allocate(0);
        event.record(stream);
        let shape = tensor.shape.as_slice();
        let shape_array = [shape[0], shape[1]];
        Ok(AsyncTensorOnCuda {
            name: "unknow_tensor".to_string(),
            ptr: data_ptr,
            shape: shape_array,
            event,
            host_ptr: pinned_host_ptr,
            dtype: TensorDtype::F16,
        })
    }
}

pub(crate) fn embedding2cuda_tensor(embedding: FeatureEmbeddingBatch) -> AsyncEmbeddingOnCuda {
    match embedding {
        FeatureEmbeddingBatch::RawEmbedding(raw_embedding) => {
            let name = raw_embedding.feature_name;
            let mut non_empty_index_list = Vec::new();

            raw_embedding
                .index
                .iter()
                .enumerate()
                .for_each(|(idx, id2idx)| {
                    if *id2idx != 0 {
                        non_empty_index_list.push(idx as u64);
                    }
                });
            AsyncEmbeddingOnCuda::Raw(AsyncRawEmbeddingOnCuda {
                tensor: cuda_tensor_h2d(PersiaDenseTensor {
                    name: name.clone(),
                    dim: raw_embedding.embeddings.shape()[1],
                    content: raw_embedding.embeddings.into_raw_vec(),
                })
                .expect("cannot move raw embedding to gpu"),
                index: cuda_tensor_h2d(PersiaDenseTensor {
                    name: name.clone(),
                    dim: raw_embedding.index.len(),
                    content: raw_embedding.index,
                })
                .expect("cannot move index to gpu"),
                non_empty_index: cuda_tensor_h2d(PersiaDenseTensor {
                    name: name.clone(),
                    dim: std::cmp::max(non_empty_index_list.len(), 1),
                    content: non_empty_index_list,
                })
                .expect("cannot move non empty list to gpu"),
                samples_id_num: raw_embedding.sample_id_num,
            })
        }
        FeatureEmbeddingBatch::SumEmbedding(sum_embedding) => {
            let name = sum_embedding.feature_name.clone();
            let embedding = cuda_tensor_h2d(PersiaDenseTensor {
                name: name.clone(),
                dim: sum_embedding.embeddings.shape()[1],
                content: sum_embedding.embeddings.into_raw_vec(),
            })
            .expect("cannot move embedding to gpu");
            AsyncEmbeddingOnCuda::Sum(embedding)
        }
    }
}
