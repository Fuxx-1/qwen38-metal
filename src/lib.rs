pub mod api;
pub mod geometry;
pub mod metal;
pub mod metal_runtime;
pub mod model;
mod mps;
pub mod mtp;
pub mod native;
pub mod paged_kv;
pub mod preflight;

pub use geometry::{CachePlan, KvPrecision, M4ProBudget, Qwen35Geometry};
pub use model::{
    inspect_mlx_safetensors_dir, open_mlx_safetensors_dir, MlxModelManifest, MlxTensor,
    MlxWeightStore,
};
pub use preflight::{inspect_model_dir, ModelInspection, MtpSupport};
