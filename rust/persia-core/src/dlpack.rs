#![allow(dead_code)]
///! For detailed documentation, see:
///! https://github.com/dmlc/dlpack
use std::os::raw::c_void;

use persia_libs::tracing;

/// DLpack DeviceType representation. Most of the scene is DLCPU and DLCUDA.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum DLDeviceType {
    DLCPU = 1,
    DLCUDA = 2,
    DLCUDAHost = 3,
    DLOpenCL = 4,
    DLVulkan = 7,
    DLMetal = 8,
    DLVPI = 9,
    DLROCM = 10,
    DLROCMHost = 11,
    DLExtDev = 12,
    DLCUDAManaged = 13,
}

/// Enum type of Dlpack DataTypeCode.
//
/// Use enum to represent the generic datatype. This struct can't infer concrete datatype directly,
/// the concrete datatype should compose with the bits field in [`DLDataType`].
#[derive(Clone, Copy)]
pub enum DLDataTypeCode {
    DLInt = 0,
    DLUInt = 1,
    DLFloat = 2,
    DLOpaqueHandle = 3,
    DLBfloat = 4,
    DLComplex = 5,
}

/// Dlpack DataType representation.It can describe almost general datatype in DeepLearning framework.
///
/// For example the [`i32`] should represent as [`DLDataType`].code=0 and [`DLDataType`].bits=4
/// The [`i64`] should represent as [`DLDataType`].code=2 and [`DLDataType`].bits=8.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

/// Dlpack Device representation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

/// Dlpack tensor format. Almost all fields are required except strides.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

impl Drop for DLTensor {
    fn drop(&mut self) {
        tracing::debug!("drop dltensor...");
    }
}

/// A wrapper of [`DLTensor`] that holds the dl_tensor data and corresponding deleter function.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
}

/// [`DLManagedTensor`] FFI C drop function
///
/// Ensure drop the instance after ownership changes.
pub extern "C" fn drop_dl_managed_tensor(drop_ptr: *mut DLManagedTensor) {
    if drop_ptr.is_null() {
        return;
    }

    unsafe { Box::from_raw(drop_ptr) };
}
