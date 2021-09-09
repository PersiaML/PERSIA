use paste::paste;

use persia_libs::{anyhow::Result, half::f16, thiserror};

#[cfg(feature = "cuda")]
use crate::cuda;

#[cfg(feature = "cuda")]
use cuda::GPUStorage;

use persia_speedy::{Readable, Writable};

// pub trait Storage_ {
//     pub fn get_dtype() -> DType;
//     pub fn data_ptr();
//     pub fn type_size();
//     pub fn device() -> String ;
// }

#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("cpu storagea not found")]
    CPUStorageNotFound,
    #[error("gpu storage not found")]
    GPUStorageNotFound,
}

#[derive(Readable, Writable, Copy, Clone, Debug)]
pub enum DType {
    F16 = 1,
    F32 = 2,
    F64 = 3,
    I32 = 4,
    I64 = 5,
    U32 = 6,
    U64 = 7,
    USIZE = 8,
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
    pub fn get_type_name(&self) -> &str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::F64 => "f64",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::USIZE => "usize",
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
            CPUStorage::F16(_) => DType::F16,
            CPUStorage::F32(_) => DType::F32,
            CPUStorage::F64(_) => DType::F64,
            CPUStorage::I32(_) => DType::I32,
            CPUStorage::I64(_) => DType::I64,
            CPUStorage::U32(_) => DType::U32,
            CPUStorage::U64(_) => DType::U64,
            CPUStorage::USIZE(_) => DType::USIZE,
        }
    }

    pub fn as_raw_ptr(&mut self) -> *const std::os::raw::c_void {
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

    pub fn data_ptr(&mut self) -> u64 {
        self.as_raw_ptr() as u64
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

    pub fn device(&self) -> String {
        match self.storage {
            Storage::CPU(_) => "cpu".to_owned(),
            #[cfg(feature = "cuda")]
            Storage::GPU(_) => "cuda".to_owned(),
        }
    }

    pub fn data_ptr(&mut self) -> u64 {
        // TODO: refactor with trait and move data_ptr is_ready field in tensor struct
        match &mut self.storage {
            Storage::CPU(val) => val.data_ptr(),
            #[cfg(feature = "cuda")]
            Storage::GPU(val) => val.data_ptr(),
        }
    }

    pub fn dtype(&self) -> DType {
        match &self.storage {
            Storage::CPU(val) => val.get_dtype(),
            #[cfg(feature = "cuda")]
            Storage::GPU(val) => val.get_dtype(),
        }
    }
}

#[derive(Readable, Writable, Debug)]
pub struct SparseTensor {
    pub data: Storage,
    pub offset: Vec<u64>,
    pub name: Option<String>,
}
