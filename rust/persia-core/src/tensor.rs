use paste::paste;

use persia_libs::{anyhow::Result, half::f16, thiserror};

#[cfg(feature = "cuda")]
use crate::cuda;

#[cfg(feature = "cuda")]
use cuda::GPUStorage;

use persia_speedy::{Readable, Writable};

#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("cpu storagea not found")]
    CPUStorageNotFound,
    #[error("gpu storage not found")]
    GPUStorageNotFound,
}

#[derive(Readable, Writable, Debug)]
pub enum DType {
    F16,
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
    USIZE,
}

impl DType {
    pub fn get_type_size(&self) -> usize {
        match self {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::F16 => std::mem::size_of::<f16>(),
            DType::F64 => std::mem::size_of::<f64>(),
            DType::I32 => std::mem::size_of::<i32>(),
            DType::I64 => std::mem::size_of::<i64>(),
            DType::U32 => std::mem::size_of::<u32>(),
            DType::U64 => std::mem::size_of::<u64>(),
            DType::USIZE => std::mem::size_of::<u64>(),
        }
    }
}

#[derive(Readable, Writable, Debug)]
pub enum CPUStorage {
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    USIZE(Vec<usize>),
}

impl CPUStorage {
    pub fn get_dtype(&self) -> DType {
        match self {
            CPUStorage::F32(_) => DType::F32,
            CPUStorage::F16(_) => DType::F16,
            CPUStorage::F64(_) => DType::F64,
            CPUStorage::I32(_) => DType::I32,
            CPUStorage::I64(_) => DType::I64,
            CPUStorage::U32(_) => DType::U32,
            CPUStorage::U64(_) => DType::U64,
            CPUStorage::USIZE(_) => DType::USIZE,
        }
    }

    pub fn as_raw_ptr(&self) -> *const std::os::raw::c_void {
        match self {
            CPUStorage::F32(val) => val.as_ptr() as *const std::os::raw::c_void,
            CPUStorage::F16(val) => val.as_ptr() as *const std::os::raw::c_void,
            CPUStorage::F64(val) => val.as_ptr() as *const std::os::raw::c_void,
            CPUStorage::I32(val) => val.as_ptr() as *const std::os::raw::c_void,
            CPUStorage::I64(val) => val.as_ptr() as *const std::os::raw::c_void,
            CPUStorage::U32(val) => val.as_ptr() as *const std::os::raw::c_void,
            CPUStorage::U64(val) => val.as_ptr() as *const std::os::raw::c_void,
            CPUStorage::USIZE(val) => val.as_ptr() as *const std::os::raw::c_void,
        }
    }
}

macro_rules! add_new_func2_cpu_storage {
    ($(($typ:ty, $attr:ident)),*) => {
        paste! {
            impl CPUStorage {
                    $(
                        pub fn [<from_ $typ:lower>](data: Vec<$typ>) -> Self {
                            CPUStorage::$attr(data)
                        }
                    )*
            }
        }
    }
}

add_new_func2_cpu_storage!(
    (f16, F16),
    (f32, F32),
    (f64, F64),
    (i32, I32),
    (i64, I64),
    (usize, USIZE),
    (u32, U32),
    (u64, U64)
);

#[derive(Readable, Writable, Debug)]
pub enum Storage {
    CPU(CPUStorage),

    #[cfg(feature = "cuda")]
    GPU(GPUStorage),
}

impl Storage {
    pub fn take_cpu_storage(self) -> Result<CPUStorage, TensorError> {
        match self {
            Storage::CPU(val) => Ok(val),
            _ => Err(TensorError::CPUStorageNotFound),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn take_gpu_storage(self) -> Result<GPUStorage, TensorError> {
        match self {
            Storage::GPU(val) => Ok(val),
            _ => Err(TensorError::GPUStorageNotFound),
        }
    }

    pub fn cpu_ref(&self) -> &CPUStorage {
        match &self {
            Storage::CPU(val) => val,
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn gpu_ref(&self) -> &GPUStorage {
        match &self {
            Storage::GPU(val) => val,
            _ => unreachable!(),
        }
    }
}
#[derive(Readable, Writable, Debug)]
pub struct Tensor {
    pub storage: Storage,
    pub shape: Vec<usize>,
    pub name: Option<String>,
}

impl Tensor {
    pub fn new(storage: Storage, shape: Vec<usize>, name: Option<String>) -> Self {
        Self {
            storage,
            shape,
            name,
        }
    }
    #[cfg(feature = "cuda")]
    pub fn cuda(self) -> Tensor {
        let shape = self.shape.clone();
        let cpu_storage = self.storage.take_cpu_storage().unwrap();
        let gpu_storage = GPUStorage::new(cpu_storage, shape).unwrap();

        Tensor {
            storage: Storage::GPU(gpu_storage),
            shape: self.shape.clone(),
            name: self.name.clone(),
        }
    }

    // #[cfg(feature = "cuda")]
    // pub fn cpu(&self) -> Result<Tensor> {}

    pub fn numpy(&self) {}
}

#[derive(Readable, Writable, Debug)]
pub struct SparseTensor {
    pub data: Storage,
    pub offset: Vec<u64>,
    pub name: Option<String>,
}
