use std::error::Error;
use std::fmt;

pub const EMBEDDED_LIBRARY: &[u8] = include_bytes!(env!("QWEN38_METALLIB"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalLibraryInfo {
    pub byte_len: usize,
}

pub fn embedded_library_info() -> Result<MetalLibraryInfo, MetalLibraryError> {
    if EMBEDDED_LIBRARY.len() < 4 || &EMBEDDED_LIBRARY[..4] != b"MTLB" {
        return Err(MetalLibraryError::InvalidMagic);
    }

    Ok(MetalLibraryInfo {
        byte_len: EMBEDDED_LIBRARY.len(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalLibraryError {
    InvalidMagic,
}

impl fmt::Display for MetalLibraryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => {
                write!(formatter, "embedded Metal library does not start with MTLB")
            }
        }
    }
}

impl Error for MetalLibraryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_library_has_a_metal_header() {
        assert!(embedded_library_info().unwrap().byte_len > 4);
    }
}
