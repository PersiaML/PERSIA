#![allow(dead_code)]
///! For detailed documentation, see:
///! https://github.com/dmlc/dlpack
use std::os::raw::c_void;

use persia_libs::tracing;

#[repr(C)]
#[derive(Clone, Copy)]
#[warn(non_camel_case_types)]
pub enum DLDeviceType {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
}

#[derive(Clone, Copy)]
#[warn(non_camel_case_types)]
pub enum DLDataTypeCode {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLOpaqueHandle = 3,
    kDLBfloat = 4,
    kDLComplex = 5,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub ndim: i32,
    pub device: DLDevice,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

impl Drop for DLTensor {
    fn drop(&mut self) {
        tracing::info!("drop dltensor...");
    }
}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
}
