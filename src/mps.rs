#![allow(unexpected_cfgs)]

use metal::{BufferRef, CommandBufferRef, DeviceRef};
use objc::rc::autoreleasepool;
use objc::runtime::{Class, Object, NO, YES};
use objc::{msg_send, sel, sel_impl};
use std::fmt;
use std::ptr::NonNull;

const MPS_DATA_TYPE_FLOAT16: u32 = 0x1000_0010;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpsError {
    ClassUnavailable(&'static str),
    AllocationFailed(&'static str),
}

impl fmt::Display for MpsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClassUnavailable(name) => write!(formatter, "MPS class {name} is unavailable"),
            Self::AllocationFailed(name) => write!(formatter, "cannot allocate MPS {name}"),
        }
    }
}

/// Retained Objective-C MPSMatrix object. The wrapper deliberately exposes
/// only the fixed FP16 matrix contract used by Q4 prompt prefill.
pub struct MpsMatrix {
    object: NonNull<Object>,
}

impl MpsMatrix {
    pub fn new_fp16(buffer: &BufferRef, rows: usize, columns: usize) -> Result<Self, MpsError> {
        let rows = u64::try_from(rows).map_err(|_| MpsError::AllocationFailed("matrix"))?;
        let columns = u64::try_from(columns).map_err(|_| MpsError::AllocationFailed("matrix"))?;
        let row_bytes = columns
            .checked_mul(2)
            .ok_or(MpsError::AllocationFailed("matrix row stride"))?;
        autoreleasepool(|| unsafe {
            let descriptor_class = class("MPSMatrixDescriptor")?;
            let descriptor: *mut Object = msg_send![descriptor_class,
                matrixDescriptorWithRows: rows
                columns: columns
                rowBytes: row_bytes
                dataType: MPS_DATA_TYPE_FLOAT16
            ];
            let matrix_class = class("MPSMatrix")?;
            let allocation: *mut Object = msg_send![matrix_class, alloc];
            let matrix: *mut Object = msg_send![allocation,
                initWithBuffer: buffer
                descriptor: descriptor
            ];
            let object = NonNull::new(matrix).ok_or(MpsError::AllocationFailed("matrix"))?;
            Ok(Self { object })
        })
    }

    fn raw(&self) -> *mut Object {
        self.object.as_ptr()
    }
}

impl Drop for MpsMatrix {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.object.as_ptr(), release];
        }
    }
}

/// Retained MPS matrix multiplication kernel for A * B-transposed. Q4 model
/// weights are stored as output rows by input columns, so transposing B lets
/// the Q4 layout remain row-major after GPU dequantization.
pub struct MpsFp16Gemm {
    object: NonNull<Object>,
}

impl MpsFp16Gemm {
    pub fn new(
        device: &DeviceRef,
        result_rows: usize,
        result_columns: usize,
        interior_columns: usize,
    ) -> Result<Self, MpsError> {
        let result_rows =
            u64::try_from(result_rows).map_err(|_| MpsError::AllocationFailed("GEMM"))?;
        let result_columns =
            u64::try_from(result_columns).map_err(|_| MpsError::AllocationFailed("GEMM"))?;
        let interior_columns =
            u64::try_from(interior_columns).map_err(|_| MpsError::AllocationFailed("GEMM"))?;
        unsafe {
            let gemm_class = class("MPSMatrixMultiplication")?;
            let allocation: *mut Object = msg_send![gemm_class, alloc];
            let gemm: *mut Object = msg_send![allocation,
                initWithDevice: device
                transposeLeft: NO
                transposeRight: YES
                resultRows: result_rows
                resultColumns: result_columns
                interiorColumns: interior_columns
                alpha: 1.0_f64
                beta: 0.0_f64
            ];
            let object = NonNull::new(gemm).ok_or(MpsError::AllocationFailed("GEMM"))?;
            Ok(Self { object })
        }
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        left: &MpsMatrix,
        right: &MpsMatrix,
        result: &MpsMatrix,
    ) {
        unsafe {
            let _: () = msg_send![self.object.as_ptr(),
                encodeToCommandBuffer: command_buffer
                leftMatrix: left.raw()
                rightMatrix: right.raw()
                resultMatrix: result.raw()
            ];
        }
    }
}

impl Drop for MpsFp16Gemm {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.object.as_ptr(), release];
        }
    }
}

pub fn is_available(device: &DeviceRef) -> bool {
    metal::mps::mps_supports_device(device)
        && Class::get("MPSMatrixDescriptor").is_some()
        && Class::get("MPSMatrix").is_some()
        && Class::get("MPSMatrixMultiplication").is_some()
}

fn class(name: &'static str) -> Result<&'static Class, MpsError> {
    Class::get(name).ok_or(MpsError::ClassUnavailable(name))
}
