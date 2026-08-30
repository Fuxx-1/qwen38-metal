use crate::geometry::{GeometryError, Qwen35Geometry};
use crate::preflight::MtpSupport;
use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

const MAX_SAFETENSORS_HEADER_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    MlxSafetensors,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MlxSafetensors => write!(formatter, "MLX safetensors"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineQuantization {
    pub bits: u8,
    pub group_size: u32,
    pub mode: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxModelManifest {
    pub format: ModelFormat,
    pub architectures: Vec<String>,
    pub model_type: String,
    pub geometry: Qwen35Geometry,
    pub quantization: AffineQuantization,
    pub shard_count: usize,
    pub indexed_tensor_count: usize,
    pub indexed_tensor_bytes: u64,
    pub quantized_tensor_groups: usize,
    pub mtp_support: MtpSupport,
}

/// A read-only, zero-copy view of MLX safetensors shards. Mapping a model does
/// not copy its tensor bytes into the Rust heap or a temporary dequantized form.
pub struct MlxWeightStore {
    shards: Vec<MappedShard>,
    tensors: BTreeMap<String, MlxTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxTensor {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub shard_index: usize,
    pub byte_offset: u64,
    pub byte_len: u64,
}

struct MappedShard {
    mapping: Mmap,
}

impl MlxWeightStore {
    pub fn tensor(&self, name: &str) -> Option<&MlxTensor> {
        self.tensors.get(name)
    }

    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        self.tensor(name)
            .and_then(|tensor| self.tensor_bytes(tensor))
    }

    pub fn tensor_bytes(&self, tensor: &MlxTensor) -> Option<&[u8]> {
        let shard = self.shards.get(tensor.shard_index)?;
        let start = usize::try_from(tensor.byte_offset).ok()?;
        let length = usize::try_from(tensor.byte_len).ok()?;
        let end = start.checked_add(length)?;
        shard.mapping.get(start..end)
    }

    pub fn tensor_values_f32(&self, name: &str) -> Result<Vec<f32>, ModelFormatError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| ModelFormatError::MissingRuntimeTensor(name.to_owned()))?;
        let bytes = self
            .tensor_bytes(tensor)
            .ok_or_else(|| ModelFormatError::InvalidRuntimeTensor(name.to_owned()))?;
        match tensor.dtype.as_str() {
            "BF16" => bytes
                .chunks_exact(2)
                .map(|chunk| {
                    Ok(f32::from_bits(
                        u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16,
                    ))
                })
                .collect(),
            "F32" => bytes
                .chunks_exact(4)
                .map(|chunk| Ok(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
                .collect(),
            dtype => Err(ModelFormatError::UnsupportedRuntimeDtype {
                name: name.to_owned(),
                dtype: dtype.to_owned(),
            }),
        }
    }

    pub fn shard_data(&self, shard_index: usize) -> Option<&[u8]> {
        self.shards
            .get(shard_index)
            .map(|shard| shard.mapping.as_ref())
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn mapped_bytes(&self) -> u64 {
        self.shards
            .iter()
            .map(|shard| shard.mapping.len() as u64)
            .sum()
    }
}

pub fn open_mlx_safetensors_dir(
    path: impl AsRef<Path>,
) -> Result<MlxWeightStore, ModelFormatError> {
    let path = path.as_ref();
    inspect_mlx_safetensors_dir(path)?;
    let index: WeightIndex = parse_json_file(
        path.join("model.safetensors.index.json"),
        JsonKind::WeightIndex,
    )?;
    let mut tensors_by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (tensor, shard) in index.weight_map {
        tensors_by_shard.entry(shard).or_default().push(tensor);
    }

    let mut shards = Vec::with_capacity(tensors_by_shard.len());
    let mut tensors = BTreeMap::new();
    for (shard_name, tensor_names) in tensors_by_shard {
        let shard_path = resolve_shard(path, &shard_name)?;
        let header = read_safetensors_header(&shard_path)?;
        let shard_index = shards.len();

        for tensor_name in tensor_names {
            let tensor = header.tensors.get(&tensor_name).ok_or_else(|| {
                ModelFormatError::MissingIndexedTensor {
                    shard: shard_path.clone(),
                    tensor: tensor_name.clone(),
                }
            })?;
            validate_tensor_range(&header, tensor, &shard_path, &tensor_name)?;
            let byte_offset = header
                .data_offset
                .checked_add(tensor.data_offsets.0)
                .ok_or(ModelFormatError::Overflow)?;
            let byte_len = tensor.data_offsets.1 - tensor.data_offsets.0;
            tensors.insert(
                tensor_name,
                MlxTensor {
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    shard_index,
                    byte_offset,
                    byte_len,
                },
            );
        }

        let file = File::open(&shard_path).map_err(|source| ModelFormatError::Read {
            path: shard_path.clone(),
            source,
        })?;
        // The safetensors files stay immutable for the lifetime of this store.
        // Mapping lets Metal later reference individual shard ranges without a heap copy.
        let mapping =
            unsafe { MmapOptions::new().map(&file) }.map_err(|source| ModelFormatError::Read {
                path: shard_path,
                source,
            })?;
        shards.push(MappedShard { mapping });
    }

    Ok(MlxWeightStore { shards, tensors })
}

pub fn inspect_mlx_safetensors_dir(
    path: impl AsRef<Path>,
) -> Result<MlxModelManifest, ModelFormatError> {
    let path = path.as_ref();
    let config: ModelConfig = parse_json_file(path.join("config.json"), JsonKind::Config)?;
    let index: WeightIndex = parse_json_file(
        path.join("model.safetensors.index.json"),
        JsonKind::WeightIndex,
    )?;
    let text = config
        .text_config
        .ok_or(ModelFormatError::MissingTextConfig)?;

    let model_type = if text.model_type.is_empty() {
        config.model_type.clone()
    } else {
        text.model_type.clone()
    };
    if !model_type.starts_with("qwen3_5") {
        return Err(ModelFormatError::UnsupportedModelType(model_type));
    }

    let full_attention_layers = count_full_attention_layers(&text);
    let geometry = Qwen35Geometry::from_parts(
        text.hidden_size,
        text.num_hidden_layers,
        text.num_key_value_heads,
        text.head_dim,
        full_attention_layers,
    )?;
    let quantization = config
        .quantization_config
        .or(config.quantization)
        .ok_or(ModelFormatError::MissingQuantization)?;
    if quantization.bits != 4 || quantization.group_size == 0 || quantization.mode != "affine" {
        return Err(ModelFormatError::UnsupportedQuantization {
            bits: quantization.bits,
            group_size: quantization.group_size,
            mode: quantization.mode,
        });
    }

    let mut tensors_by_shard: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for (tensor, shard) in &index.weight_map {
        tensors_by_shard
            .entry(shard.clone())
            .or_default()
            .push(tensor.as_str());
    }

    let mut indexed_tensor_bytes = 0_u64;
    for (shard_name, tensors) in &tensors_by_shard {
        let shard_path = resolve_shard(path, shard_name)?;
        let header = read_safetensors_header(&shard_path)?;

        for tensor_name in tensors {
            let tensor = header.tensors.get(*tensor_name).ok_or_else(|| {
                ModelFormatError::MissingIndexedTensor {
                    shard: shard_path.clone(),
                    tensor: (*tensor_name).to_owned(),
                }
            })?;
            validate_tensor_range(&header, tensor, &shard_path, tensor_name)?;
            indexed_tensor_bytes = indexed_tensor_bytes
                .checked_add(tensor.data_offsets.1 - tensor.data_offsets.0)
                .ok_or(ModelFormatError::Overflow)?;
        }
    }

    let quantized_tensor_groups = validate_affine_groups(&index.weight_map)?;
    let mtp_tensor_count = index
        .weight_map
        .keys()
        .filter(|tensor| is_mtp_tensor(tensor))
        .count();
    let mtp_support = mtp_support(text.mtp_num_hidden_layers, mtp_tensor_count);

    Ok(MlxModelManifest {
        format: ModelFormat::MlxSafetensors,
        architectures: config.architectures,
        model_type,
        geometry,
        quantization: AffineQuantization {
            bits: quantization.bits,
            group_size: quantization.group_size,
            mode: quantization.mode,
        },
        shard_count: tensors_by_shard.len(),
        indexed_tensor_count: index.weight_map.len(),
        indexed_tensor_bytes,
        quantized_tensor_groups,
        mtp_support,
    })
}

fn count_full_attention_layers(text: &TextConfig) -> u32 {
    if text.layer_types.is_empty() {
        let interval = text.full_attention_interval.max(1);
        text.num_hidden_layers.div_ceil(interval)
    } else {
        text.layer_types
            .iter()
            .filter(|layer| layer.as_str() == "full_attention")
            .count() as u32
    }
}

fn resolve_shard(root: &Path, shard_name: &str) -> Result<PathBuf, ModelFormatError> {
    let shard = Path::new(shard_name);
    if shard.is_absolute()
        || shard
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ModelFormatError::UnsafeShardPath(shard_name.to_owned()));
    }

    Ok(root.join(shard))
}

fn parse_json_file<T: for<'de> Deserialize<'de>>(
    path: PathBuf,
    kind: JsonKind,
) -> Result<T, ModelFormatError> {
    let contents = fs::read_to_string(&path).map_err(|source| ModelFormatError::Read {
        path: path.clone(),
        source,
    })?;
    serde_json::from_str(&contents).map_err(|source| ModelFormatError::Json { path, kind, source })
}

fn read_safetensors_header(path: &Path) -> Result<SafetensorsHeader, ModelFormatError> {
    let mut file = File::open(path).map_err(|source| ModelFormatError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let file_bytes = file
        .metadata()
        .map_err(|source| ModelFormatError::Read {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if file_bytes < 8 {
        return Err(ModelFormatError::InvalidHeaderLength {
            path: path.to_path_buf(),
            header_bytes: 0,
            file_bytes,
        });
    }

    let mut encoded_length = [0_u8; 8];
    file.read_exact(&mut encoded_length)
        .map_err(|source| ModelFormatError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let header_bytes = u64::from_le_bytes(encoded_length);
    if header_bytes > MAX_SAFETENSORS_HEADER_BYTES || header_bytes > file_bytes - 8 {
        return Err(ModelFormatError::InvalidHeaderLength {
            path: path.to_path_buf(),
            header_bytes,
            file_bytes,
        });
    }

    let mut serialized = vec![0_u8; header_bytes as usize];
    file.read_exact(&mut serialized)
        .map_err(|source| ModelFormatError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let values: BTreeMap<String, Value> =
        serde_json::from_slice(&serialized).map_err(|source| ModelFormatError::Json {
            path: path.to_path_buf(),
            kind: JsonKind::SafetensorsHeader,
            source,
        })?;

    let mut tensors = BTreeMap::new();
    for (name, value) in values {
        if name == "__metadata__" {
            continue;
        }
        let tensor: SafetensorsTensor =
            serde_json::from_value(value).map_err(|source| ModelFormatError::Json {
                path: path.to_path_buf(),
                kind: JsonKind::SafetensorsHeader,
                source,
            })?;
        if tensor.dtype.is_empty() || tensor.shape.is_empty() {
            return Err(ModelFormatError::InvalidTensorMetadata {
                path: path.to_path_buf(),
                tensor: name,
            });
        }
        tensors.insert(
            name,
            TensorHeader {
                dtype: tensor.dtype,
                shape: tensor.shape,
                data_offsets: (tensor.data_offsets[0], tensor.data_offsets[1]),
            },
        );
    }

    Ok(SafetensorsHeader {
        tensors,
        data_bytes: file_bytes - 8 - header_bytes,
        data_offset: 8 + header_bytes,
    })
}

fn validate_tensor_range(
    header: &SafetensorsHeader,
    tensor: &TensorHeader,
    shard: &Path,
    tensor_name: &str,
) -> Result<(), ModelFormatError> {
    if tensor.data_offsets.0 > tensor.data_offsets.1 || tensor.data_offsets.1 > header.data_bytes {
        return Err(ModelFormatError::InvalidTensorOffsets {
            shard: shard.to_path_buf(),
            tensor: tensor_name.to_owned(),
        });
    }
    Ok(())
}

fn validate_affine_groups(
    weight_map: &BTreeMap<String, String>,
) -> Result<usize, ModelFormatError> {
    let mut groups = 0;
    for tensor in weight_map
        .keys()
        .filter(|tensor| tensor.ends_with(".weight"))
    {
        let base = tensor.strip_suffix(".weight").expect("suffix was checked");
        let scales = format!("{base}.scales");
        let biases = format!("{base}.biases");
        let has_scales = weight_map.contains_key(&scales);
        let has_biases = weight_map.contains_key(&biases);

        if has_scales != has_biases {
            return Err(ModelFormatError::IncompleteAffineGroup {
                weight: tensor.to_owned(),
                missing: if has_scales { "biases" } else { "scales" },
            });
        }
        if has_scales {
            groups += 1;
        }
    }

    Ok(groups)
}

fn is_mtp_tensor(name: &str) -> bool {
    name.split('.')
        .any(|segment| segment.eq_ignore_ascii_case("mtp"))
        || name.contains("nextn")
        || name.contains("next_n")
}

fn mtp_support(configured_layers: u32, tensor_count: usize) -> MtpSupport {
    match (configured_layers, tensor_count) {
        (0, _) => MtpSupport::NotDeclared,
        (configured_layers, 0) => MtpSupport::DeclaredButWeightsMissing { configured_layers },
        (configured_layers, tensor_count) => MtpSupport::Available {
            configured_layers,
            tensor_count,
        },
    }
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    text_config: Option<TextConfig>,
    #[serde(default)]
    quantization: Option<QuantizationConfig>,
    #[serde(default)]
    quantization_config: Option<QuantizationConfig>,
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
}

#[derive(Debug, Deserialize)]
struct QuantizationConfig {
    bits: u8,
    group_size: u32,
    mode: String,
}

#[derive(Debug, Deserialize)]
struct WeightIndex {
    #[serde(default)]
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug)]
struct SafetensorsHeader {
    tensors: BTreeMap<String, TensorHeader>,
    data_bytes: u64,
    data_offset: u64,
}

#[derive(Debug)]
struct TensorHeader {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

#[derive(Debug, Clone, Copy)]
pub enum JsonKind {
    Config,
    WeightIndex,
    SafetensorsHeader,
}

impl fmt::Display for JsonKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config => write!(formatter, "config.json"),
            Self::WeightIndex => write!(formatter, "model.safetensors.index.json"),
            Self::SafetensorsHeader => write!(formatter, "safetensors header"),
        }
    }
}

#[derive(Debug)]
pub enum ModelFormatError {
    Read {
        path: PathBuf,
        source: io::Error,
    },
    Json {
        path: PathBuf,
        kind: JsonKind,
        source: serde_json::Error,
    },
    MissingTextConfig,
    UnsupportedModelType(String),
    MissingQuantization,
    UnsupportedQuantization {
        bits: u8,
        group_size: u32,
        mode: String,
    },
    UnsafeShardPath(String),
    InvalidHeaderLength {
        path: PathBuf,
        header_bytes: u64,
        file_bytes: u64,
    },
    MissingIndexedTensor {
        shard: PathBuf,
        tensor: String,
    },
    InvalidTensorOffsets {
        shard: PathBuf,
        tensor: String,
    },
    InvalidTensorMetadata {
        path: PathBuf,
        tensor: String,
    },
    MissingRuntimeTensor(String),
    InvalidRuntimeTensor(String),
    UnsupportedRuntimeDtype {
        name: String,
        dtype: String,
    },
    IncompleteAffineGroup {
        weight: String,
        missing: &'static str,
    },
    Geometry(GeometryError),
    Overflow,
}

impl fmt::Display for ModelFormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => write!(formatter, "cannot read {}: {source}", path.display()),
            Self::Json { path, kind, source } => {
                write!(formatter, "cannot parse {kind} at {}: {source}", path.display())
            }
            Self::MissingTextConfig => write!(formatter, "config.json does not contain text_config"),
            Self::UnsupportedModelType(model_type) => {
                write!(formatter, "unsupported model type {model_type:?}; expected qwen3_5")
            }
            Self::MissingQuantization => write!(formatter, "model config does not declare MLX quantization"),
            Self::UnsupportedQuantization {
                bits,
                group_size,
                mode,
            } => write!(
                formatter,
                "unsupported quantization bits={bits}, group_size={group_size}, mode={mode:?}; expected 4-bit affine"
            ),
            Self::UnsafeShardPath(shard) => {
                write!(formatter, "weight index contains an unsafe shard path {shard:?}")
            }
            Self::InvalidHeaderLength {
                path,
                header_bytes,
                file_bytes,
            } => write!(
                formatter,
                "invalid safetensors header length {header_bytes} for {} ({file_bytes} bytes)",
                path.display()
            ),
            Self::MissingIndexedTensor { shard, tensor } => write!(
                formatter,
                "weight index tensor {tensor:?} is absent from {}",
                shard.display()
            ),
            Self::InvalidTensorOffsets { shard, tensor } => write!(
                formatter,
                "tensor {tensor:?} has offsets outside the data section of {}",
                shard.display()
            ),
            Self::InvalidTensorMetadata { path, tensor } => write!(
                formatter,
                "tensor {tensor:?} has invalid metadata in {}",
                path.display()
            ),
            Self::MissingRuntimeTensor(name) => write!(formatter, "runtime tensor {name:?} is absent"),
            Self::InvalidRuntimeTensor(name) => write!(formatter, "runtime tensor {name:?} has an invalid byte range"),
            Self::UnsupportedRuntimeDtype { name, dtype } => write!(
                formatter,
                "runtime tensor {name:?} uses unsupported dtype {dtype:?}; expected BF16 or F32"
            ),
            Self::IncompleteAffineGroup { weight, missing } => write!(
                formatter,
                "quantized tensor {weight:?} is missing its affine {missing} companion"
            ),
            Self::Geometry(error) => write!(formatter, "invalid Qwen geometry: {error}"),
            Self::Overflow => write!(formatter, "model tensor size overflows u64"),
        }
    }
}

impl Error for ModelFormatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Geometry(error) => Some(error),
            Self::MissingTextConfig
            | Self::UnsupportedModelType(_)
            | Self::MissingQuantization
            | Self::UnsupportedQuantization { .. }
            | Self::UnsafeShardPath(_)
            | Self::InvalidHeaderLength { .. }
            | Self::MissingIndexedTensor { .. }
            | Self::InvalidTensorOffsets { .. }
            | Self::InvalidTensorMetadata { .. }
            | Self::MissingRuntimeTensor(_)
            | Self::InvalidRuntimeTensor(_)
            | Self::UnsupportedRuntimeDtype { .. }
            | Self::IncompleteAffineGroup { .. }
            | Self::Overflow => None,
        }
    }
}

impl From<GeometryError> for ModelFormatError {
    fn from(error: GeometryError) -> Self {
        Self::Geometry(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn reads_an_mlx_affine_manifest_without_loading_tensor_data() {
        let directory = fixture_dir("valid");
        write_fixture_config(&directory);
        write_fixture_index(
            &directory,
            json!({
                "language_model.model.layers.0.mlp.up_proj.weight": "model.safetensors",
                "language_model.model.layers.0.mlp.up_proj.scales": "model.safetensors",
                "language_model.model.layers.0.mlp.up_proj.biases": "model.safetensors"
            }),
        );
        write_fixture_shard(&directory.join("model.safetensors"));

        let manifest = inspect_mlx_safetensors_dir(&directory).unwrap();

        assert_eq!(manifest.format, ModelFormat::MlxSafetensors);
        assert_eq!(manifest.shard_count, 1);
        assert_eq!(manifest.indexed_tensor_count, 3);
        assert_eq!(manifest.indexed_tensor_bytes, 12);
        assert_eq!(manifest.quantized_tensor_groups, 1);
        assert_eq!(manifest.geometry.full_attention_layers, 1);
        assert!(matches!(
            manifest.mtp_support,
            MtpSupport::DeclaredButWeightsMissing {
                configured_layers: 1
            }
        ));

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn maps_safetensors_tensor_ranges_without_copying_them_to_the_heap() {
        let directory = fixture_dir("mapped-store");
        write_fixture_config(&directory);
        write_fixture_index(
            &directory,
            json!({
                "language_model.model.layers.0.mlp.up_proj.weight": "model.safetensors",
                "language_model.model.layers.0.mlp.up_proj.scales": "model.safetensors",
                "language_model.model.layers.0.mlp.up_proj.biases": "model.safetensors"
            }),
        );
        write_fixture_shard(&directory.join("model.safetensors"));

        let store = open_mlx_safetensors_dir(&directory).unwrap();
        let tensor = store
            .tensor("language_model.model.layers.0.mlp.up_proj.weight")
            .unwrap();

        assert_eq!(store.shard_count(), 1);
        assert_eq!(store.tensor_count(), 3);
        assert_eq!(tensor.dtype, "U8");
        assert_eq!(tensor.shape, vec![4]);
        assert_eq!(store.tensor_bytes(tensor), Some(&[0_u8; 4][..]));
        assert!(store.mapped_bytes() > tensor.byte_len);

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_a_quantized_weight_with_a_missing_affine_companion() {
        let directory = fixture_dir("missing-affine");
        write_fixture_config(&directory);
        write_fixture_index(
            &directory,
            json!({
                "language_model.model.layers.0.mlp.up_proj.weight": "model.safetensors",
                "language_model.model.layers.0.mlp.up_proj.scales": "model.safetensors"
            }),
        );
        write_fixture_shard(&directory.join("model.safetensors"));

        let error = inspect_mlx_safetensors_dir(&directory).unwrap_err();

        assert!(matches!(
            error,
            ModelFormatError::IncompleteAffineGroup {
                missing: "biases",
                ..
            }
        ));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_an_index_that_tries_to_escape_the_model_directory() {
        let directory = fixture_dir("unsafe-shard");
        write_fixture_config(&directory);
        write_fixture_index(
            &directory,
            json!({
                "language_model.model.layers.0.mlp.up_proj.weight": "../outside.safetensors"
            }),
        );

        let error = inspect_mlx_safetensors_dir(&directory).unwrap_err();

        assert!(matches!(error, ModelFormatError::UnsafeShardPath(_)));
        fs::remove_dir_all(directory).unwrap();
    }

    fn fixture_dir(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "qwen38-metal-model-test-{label}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn write_fixture_config(directory: &Path) {
        fs::write(
            directory.join("config.json"),
            r#"{
              "architectures": ["Qwen3_5ForConditionalGeneration"],
              "model_type": "qwen3_5",
              "quantization_config": {"bits": 4, "group_size": 64, "mode": "affine"},
              "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "num_hidden_layers": 4,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "mtp_num_hidden_layers": 1
              }
            }"#,
        )
        .unwrap();
    }

    fn write_fixture_index(directory: &Path, weight_map: Value) {
        fs::write(
            directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
    }

    fn write_fixture_shard(path: &Path) {
        let header = serde_json::to_vec(&json!({
            "language_model.model.layers.0.mlp.up_proj.weight": {
                "dtype": "U8",
                "shape": [4],
                "data_offsets": [0, 4]
            },
            "language_model.model.layers.0.mlp.up_proj.scales": {
                "dtype": "F16",
                "shape": [2],
                "data_offsets": [4, 8]
            },
            "language_model.model.layers.0.mlp.up_proj.biases": {
                "dtype": "F16",
                "shape": [2],
                "data_offsets": [8, 12]
            }
        }))
        .unwrap();
        let mut contents = (header.len() as u64).to_le_bytes().to_vec();
        contents.extend(header);
        contents.extend([0_u8; 12]);
        fs::write(path, contents).unwrap();
    }
}
