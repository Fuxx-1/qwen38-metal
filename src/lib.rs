pub mod geometry;
pub mod metal;
pub mod mtp;
pub mod paged_kv;
pub mod preflight;

pub use geometry::{CachePlan, KvPrecision, M4ProBudget, Qwen35Geometry};
pub use preflight::{inspect_model_dir, ModelInspection, MtpSupport};
