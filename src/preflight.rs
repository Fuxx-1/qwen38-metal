use crate::geometry::{GeometryError, Qwen35Geometry};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInspection {
    pub architectures: Vec<String>,
    pub max_context_tokens: Option<u32>,
    pub geometry: Qwen35Geometry,
    pub mtp_support: MtpSupport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MtpSupport {
    NotDeclared,
    DeclaredButWeightsMissing {
        configured_layers: u32,
    },
    Available {
        configured_layers: u32,
        tensor_count: usize,
    },
}

impl MtpSupport {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }
}

impl fmt::Display for MtpSupport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotDeclared => write!(formatter, "not declared by config"),
            Self::DeclaredButWeightsMissing { configured_layers } => write!(
                formatter,
                "declared ({configured_layers} layer(s)) but matching tensors are absent"
            ),
            Self::Available {
                configured_layers,
                tensor_count,
            } => write!(
                formatter,
                "available ({configured_layers} layer(s), {tensor_count} MTP tensors)"
            ),
        }
    }
}

pub fn inspect_model_dir(path: impl AsRef<Path>) -> Result<ModelInspection, PreflightError> {
    let path = path.as_ref();
    let config_path = path.join("config.json");
    let index_path = path.join("model.safetensors.index.json");
    let config_json = read_file(&config_path)?;
    let index_json = read_file(&index_path)?;

    inspect_json(&config_json, &index_json)
}

pub fn inspect_json(
    config_json: &str,
    weight_index_json: &str,
) -> Result<ModelInspection, PreflightError> {
    let config: ModelConfig =
        serde_json::from_str(config_json).map_err(PreflightError::ConfigJson)?;
    let index: WeightIndex =
        serde_json::from_str(weight_index_json).map_err(PreflightError::WeightIndexJson)?;
    let text = config
        .text_config
        .ok_or(PreflightError::MissingTextConfig)?;

    let model_type = if text.model_type.is_empty() {
        config.model_type.as_str()
    } else {
        text.model_type.as_str()
    };
    if !model_type.starts_with("qwen3_5") {
        return Err(PreflightError::UnsupportedModelType(model_type.to_owned()));
    }

    let full_attention_layers = if text.layer_types.is_empty() {
        let interval = if text.full_attention_interval == 0 {
            4
        } else {
            text.full_attention_interval
        };
        text.num_hidden_layers.div_ceil(interval)
    } else {
        text.layer_types
            .iter()
            .filter(|layer| layer.as_str() == "full_attention")
            .count() as u32
    };

    let geometry = Qwen35Geometry::from_parts(
        text.hidden_size,
        text.num_hidden_layers,
        text.num_key_value_heads,
        text.head_dim,
        full_attention_layers,
    )?;

    let mtp_tensor_count = index
        .weight_map
        .keys()
        .filter(|name| is_mtp_tensor(name))
        .count();
    let mtp_support = match (text.mtp_num_hidden_layers, mtp_tensor_count) {
        (0, _) => MtpSupport::NotDeclared,
        (configured_layers, 0) => MtpSupport::DeclaredButWeightsMissing { configured_layers },
        (configured_layers, tensor_count) => MtpSupport::Available {
            configured_layers,
            tensor_count,
        },
    };

    Ok(ModelInspection {
        architectures: config.architectures,
        max_context_tokens: non_zero(text.max_position_embeddings),
        geometry,
        mtp_support,
    })
}

fn is_mtp_tensor(name: &str) -> bool {
    name.split('.')
        .any(|segment| segment.eq_ignore_ascii_case("mtp"))
        || name.contains("nextn")
        || name.contains("next_n")
}

fn non_zero(value: u32) -> Option<u32> {
    (value != 0).then_some(value)
}

fn read_file(path: &Path) -> Result<String, PreflightError> {
    fs::read_to_string(path).map_err(|source| PreflightError::Read {
        path: path.to_path_buf(),
        source,
    })
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    text_config: Option<TextConfig>,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    #[serde(default)]
    model_type: String,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    #[serde(default)]
    full_attention_interval: u32,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default)]
    mtp_num_hidden_layers: u32,
    #[serde(default)]
    max_position_embeddings: u32,
}

#[derive(Debug, Deserialize)]
struct WeightIndex {
    #[serde(default)]
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug)]
pub enum PreflightError {
    Read { path: PathBuf, source: io::Error },
    ConfigJson(serde_json::Error),
    WeightIndexJson(serde_json::Error),
    MissingTextConfig,
    UnsupportedModelType(String),
    Geometry(GeometryError),
}

impl fmt::Display for PreflightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(formatter, "cannot read {}: {source}", path.display())
            }
            Self::ConfigJson(error) => write!(formatter, "cannot parse config.json: {error}"),
            Self::WeightIndexJson(error) => {
                write!(formatter, "cannot parse model.safetensors.index.json: {error}")
            }
            Self::MissingTextConfig => {
                write!(formatter, "config.json does not contain text_config")
            }
            Self::UnsupportedModelType(model_type) => {
                write!(formatter, "unsupported model type {model_type:?}; expected qwen3_5")
            }
            Self::Geometry(error) => write!(formatter, "invalid Qwen geometry: {error}"),
        }
    }
}

impl Error for PreflightError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::ConfigJson(error) | Self::WeightIndexJson(error) => Some(error),
            Self::Geometry(error) => Some(error),
            Self::MissingTextConfig | Self::UnsupportedModelType(_) => None,
        }
    }
}

impl From<GeometryError> for PreflightError {
    fn from(error: GeometryError) -> Self {
        Self::Geometry(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
    {
      "architectures": ["Qwen3_5ForConditionalGeneration"],
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
        "mtp_num_hidden_layers": 1,
        "max_position_embeddings": 262144
      }
    }
    "#;

    #[test]
    fn detects_declared_mtp_when_conversion_dropped_its_weights() {
        let inspection = inspect_json(
            CONFIG,
            r#"{"weight_map":{"language_model.model.layers.0.mlp.up_proj.weight":"model.safetensors"}}"#,
        )
        .unwrap();

        assert_eq!(inspection.geometry.full_attention_layers, 2);
        assert!(matches!(
            inspection.mtp_support,
            MtpSupport::DeclaredButWeightsMissing { configured_layers: 1 }
        ));
    }

    #[test]
    fn detects_matching_mtp_weights() {
        let inspection = inspect_json(
            CONFIG,
            r#"{"weight_map":{"language_model.model.mtp.layers.0.weight":"model.safetensors"}}"#,
        )
        .unwrap();

        assert!(inspection.mtp_support.is_available());
    }
}
