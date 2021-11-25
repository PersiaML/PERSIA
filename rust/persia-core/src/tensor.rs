//! This library implement the basic [`Tensor`] struct From help of [`DLTensor`].It can accept the data from different device and datatype.

use std::fmt;

use persia_libs::{anyhow::Result, half::f16, thiserror, tracing};

#[cfg(feature = "cuda")]
use crate::cuda::GPUStorage;
use crate::dlpack::*;

use persia_speedy::{Readable, Writable};

/// TensorError cover the error of [`Tensor`].
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("cpu storagea not found")]
    CPUStorageNotFound,
    #[error("gpu storage not found")]
    GPUStorageNotFound,
}

/// Enum representation of rust datatype.
#[derive(Readable, Writable, Copy, Clone, Debug)]
pub enum DTypeImpl {
    F16 = 1,
    F32 = 2,
    F64 = 3,
    I8 = 4,
    I16 = 5,
    I32 = 6,
    I64 = 7,
    U8 = 8,
    U16 = 9,
    U32 = 10,
    U64 = 11,
    BOOL = 12,
}

impl fmt::Display for DTypeImpl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl DTypeImpl {
    /// Bit size of current datatype.
    pub fn get_type_size(&self) -> usize {
        match self {
            DTypeImpl::BOOL => std::mem::size_of::<bool>(),
            DTypeImpl::F16 => std::mem::size_of::<f16>(),
            DTypeImpl::F32 => std::mem::size_of::<f32>(),
            DTypeImpl::F64 => std::mem::size_of::<f64>(),
            DTypeImpl::I8 => std::mem::size_of::<i8>(),
            DTypeImpl::I16 => std::mem::size_of::<i16>(),
            DTypeImpl::I32 => std::mem::size_of::<i32>(),
            DTypeImpl::I64 => std::mem::size_of::<i64>(),
            DTypeImpl::U8 => std::mem::size_of::<u8>(),
            DTypeImpl::U16 => std::mem::size_of::<u16>(),
            DTypeImpl::U32 => std::mem::size_of::<u32>(),
            DTypeImpl::U64 => std::mem::size_of::<u64>(),
        }
    }

    /// Name of current datatype
    pub fn get_type_name(&self) -> String {
        self.to_string()
    }

    /// Convert to [`DLDataType`].
    pub fn to_dldtype(&self) -> DLDataType {
        let (code, bits) = match self {
            DTypeImpl::F16 => (*&DLDataTypeCode::DLFloat, 16),
            DTypeImpl::F32 => (*&DLDataTypeCode::DLFloat, 32),
            DTypeImpl::F64 => (*&DLDataTypeCode::DLFloat, 64),
            DTypeImpl::I8 => (*&DLDataTypeCode::DLInt, 8),
            DTypeImpl::I16 => (*&DLDataTypeCode::DLInt, 16),
            DTypeImpl::I32 => (*&DLDataTypeCode::DLInt, 32),
            DTypeImpl::I64 => (*&DLDataTypeCode::DLInt, 64),
            DTypeImpl::U8 => (*&DLDataTypeCode::DLUInt, 8),
            DTypeImpl::BOOL => (*&DLDataTypeCode::DLUInt, 8),
            _ => panic!(
                "converting {} to DLDataType failed, uint8 is the only supported unsigned integer type in PyTorch",
                self
            )
        };
        let code = code as u8;

        DLDataType {
            code,
            bits,
            lanes: 1,
        }
    }
}

/// Storarge that store the vector data.
#[derive(Readable, Writable, Debug)]
pub enum CPUStorage {
    BOOL(Vec<bool>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl CPUStorage {
    pub fn get_dtype(&self) -> DTypeImpl {
        match self {
            CPUStorage::F16(_) => DTypeImpl::F16,
            CPUStorage::F32(_) => DTypeImpl::F32,
            CPUStorage::F64(_) => DTypeImpl::F64,
            CPUStorage::I8(_) => DTypeImpl::I8,
            CPUStorage::I16(_) => DTypeImpl::I16,
            CPUStorage::I32(_) => DTypeImpl::I32,
            CPUStorage::I64(_) => DTypeImpl::I64,
            CPUStorage::U8(_) => DTypeImpl::U8,
            CPUStorage::U16(_) => DTypeImpl::U16,
            CPUStorage::U32(_) => DTypeImpl::U32,
            CPUStorage::U64(_) => DTypeImpl::U64,
            CPUStorage::BOOL(_) => DTypeImpl::BOOL,
        }
    }

    pub fn get_raw_ptr(&mut self) -> *mut std::os::raw::c_void {
        match self {
            CPUStorage::BOOL(val) => val.as_ptr() as *mut std::os::raw::c_void,
            CPUStorage::F16(val) => val.as_ptr() as *mut std::os::raw::c_void,
            CPUStorage::F32(val) => val.as_ptr() as *mut std::os::raw::c_void,
            CPUStorage::F64(val) => val.as_ptr() as *mut std::os::raw::c_void,
            CPUStorage::I8(val) => val.as_ptr() as *mut std::os::raw::c_void,
            CPUStorage::I16(val) => val.as_ptr() as *mut std::os::raw::c_void,
            CPUStorage::I32(val) => val.as_ptr() as *mut std::os::raw::c_void,
            CPUStorage::I64(val) => val.as_ptr() as *mut std::os::raw::c_void,
            CPUStorage::U8(val) => val.as_ptr() as *mut std::os::raw::c_void,
            CPUStorage::U16(val) => val.as_ptr() as *mut std::os::raw::c_void,
            CPUStorage::U32(val) => val.as_ptr() as *mut std::os::raw::c_void,
            CPUStorage::U64(val) => val.as_ptr() as *mut std::os::raw::c_void,
        }
    }

    pub fn data_ptr(&mut self) -> u64 {
        self.get_raw_ptr() as u64
    }
}

#[derive(Readable, Writable, Debug)]
pub enum Storage {
    CPU(CPUStorage),

    #[cfg(feature = "cuda")]
    GPU(GPUStorage),
}

impl Storage {
    pub fn consume_cpu_storage(self) -> Result<CPUStorage, TensorError> {
        match self {
            Storage::CPU(val) => Ok(val),
            _ => Err(TensorError::CPUStorageNotFound),
        }
    }
}

#[derive(Readable, Writable, Debug)]
pub enum DeviceType {
    CPU,
    GPU,
}

#[derive(Readable, Writable, Debug)]
pub struct Device {
    device_type: DeviceType,
    device_id: Option<i32>,
}

impl Device {
    fn with_device_id(device_id: Option<i32>) -> Self {
        match &device_id {
            Some(device_id) => Device {
                device_type: DeviceType::GPU,
                device_id: Some(*device_id),
            },
            None => Device {
                device_type: DeviceType::CPU,
                device_id: None,
            },
        }
    }

    fn to_dldevicetype(&self) -> DLDevice {
        match self.device_type {
            DeviceType::CPU => DLDevice {
                device_id: 0i32,
                device_type: DLDeviceType::DLCPU,
            },
            DeviceType::GPU => DLDevice {
                device_id: *self.device_id.as_ref().unwrap(),
                device_type: DLDeviceType::DLCUDA,
            },
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Device {
            device_type: DeviceType::CPU,
            device_id: None,
        }
    }
}

pub fn get_stride_by_shape(shape: &[usize]) -> Vec<i64> {
    let dim = shape.len();
    let mut result = vec![1i64; dim];

    for i in 1..dim {
        result[dim - i - 1] = result[dim - i] * shape[dim - i] as i64;
    }
    result
}

#[derive(Readable, Writable, Debug)]
pub struct TensorImpl {
    pub storage: Storage,
    pub shape: Vec<usize>,
    pub stride: Vec<i64>,
    pub name: Option<String>,
    pub device: Device,
}

impl TensorImpl {
    pub fn new(
        storage: Storage,
        shape: Vec<usize>,
        name: Option<String>,
        device_id: Option<i32>,
    ) -> Self {
        let stride = get_stride_by_shape(shape.as_slice());
        let device = Device::with_device_id(device_id);

        Self {
            storage,
            shape,
            stride,
            name,
            device,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn to(self, device: &Option<i32>) -> TensorImpl {
        if let Some(device_id) = device {
            self.cuda(*device_id)
        } else {
            self
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn to(self, _device: &Option<i32>) -> TensorImpl {
        self
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(self, device_id: i32) -> TensorImpl {
        let shape = self.shape.clone();
        let cpu_storage = self.storage.consume_cpu_storage().unwrap();
        let gpu_storage = GPUStorage::new(cpu_storage, shape).unwrap();

        TensorImpl {
            storage: Storage::GPU(gpu_storage),
            shape: self.shape,
            stride: self.stride,
            name: self.name,
            device: Device::with_device_id(Some(device_id)),
        }
    }

    pub fn device(&self) -> String {
        match self.storage {
            Storage::CPU(_) => "cpu".to_owned(),
            #[cfg(feature = "cuda")]
            Storage::GPU(_) => "cuda".to_owned(),
        }
    }

    pub fn raw_data_ptr(&mut self) -> *mut std::os::raw::c_void {
        match &mut self.storage {
            Storage::CPU(val) => val.get_raw_ptr(),
            #[cfg(feature = "cuda")]
            Storage::GPU(val) => val.get_raw_ptr(),
        }
    }

    pub fn data_ptr(&mut self) -> u64 {
        self.raw_data_ptr() as u64
    }

    pub fn dtype(&self) -> DTypeImpl {
        match &self.storage {
            Storage::CPU(val) => val.get_dtype(),
            #[cfg(feature = "cuda")]
            Storage::GPU(val) => val.get_dtype(),
        }
    }

    pub fn dlpack(&mut self) -> DLManagedTensor {
        let dl_tensor = DLTensor {
            data: self.raw_data_ptr(),
            device: self.device.to_dldevicetype(),
            ndim: self.shape.len() as i32,
            dtype: self.dtype().to_dldtype(),
            shape: self.shape.as_mut_ptr() as *mut i64,
            strides: self.stride.as_mut_ptr(),
            byte_offset: 0u64,
        };
        tracing::debug!(
            "dltensor device dtype is {:?}, shape is {:?}, strides is {:?}",
            &dl_tensor.device,
            &self.shape,
            &self.stride
        );
        DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(drop_dl_managed_tensor),
        }
    }
}
