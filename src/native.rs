use crate::api::{
    parse_model_output, EngineError, ExecutionKind, FinishReason, Generation, GenerationEvent,
    GenerationRequest, InferenceEngine, InputImage, ModelDescriptor, PromptPart, PromptRole,
    ToolChoice,
};
use crate::metal_runtime::{
    DeltaNetConfig, MappedQ4AffineJob, MappedWeightBuffers, MetalBatchDecodeLayer,
    MetalBatchDecodeState, MetalDecodeFullLayer, MetalDecodeLayer, MetalDecodeLinearLayer,
    MetalDecodeState, MetalDeltaNetSnapshots, MetalDeltaNetState, MetalDeltaNetWeights,
    MetalF32Buffer, MetalGqaDecodeConfig, MetalMtpMlpF16, MetalMtpVerifyResult, MetalRuntime,
    MetalRuntimeError, Q8KvState,
};
use crate::model::{open_mlx_safetensors_dir, MlxTensor, MlxWeightStore};
use crate::mtp::{accepted_token_count, MtpController, SpeculativeDecodeSupport};
use crate::preflight::{inspect_model_dir, MtpSupport};
use image::imageops::FilterType;
use serde::Deserialize;
use serde_json::json;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const VALUES_PER_PACKED_WORD: u64 = 8;
const AFFINE_GROUP_SIZE: u64 = 64;
const DEFAULT_EOS_TOKEN_ID: u32 = 248_044;
const END_OF_MESSAGE_TOKEN_ID: u32 = 248_046;
const DEFAULT_PREFIX_CACHE_ENTRIES: usize = 2;
const DEFAULT_PREFIX_CACHE_TOKENS: usize = 65_536;
const DEFAULT_PREFIX_CACHE_MIN_TOKENS: usize = 64;
const DEFAULT_MTP_DRAFT_TOKENS: usize = 1;
// Keeps MPS matrix work large enough to amortize Q4 expansion without
// materializing an entire long prompt's MLP intermediates at once.
const PREFILL_CHUNK_TOKENS: usize = 8_192;

#[derive(Debug, Deserialize)]
struct RuntimeConfig {
    #[serde(default)]
    text_config: Option<TextRuntimeConfig>,
    #[serde(default)]
    vision_config: Option<VisionRuntimeConfig>,
    #[serde(default)]
    image_token_id: Option<u32>,
    #[serde(default)]
    vision_start_token_id: Option<u32>,
    #[serde(default)]
    vision_end_token_id: Option<u32>,
}

#[derive(Debug, Deserialize, Clone)]
struct VisionRuntimeConfig {
    depth: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_position_embeddings: usize,
    out_hidden_size: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    in_channels: usize,
}

#[derive(Debug, Deserialize, Default)]
struct GenerationRuntimeConfig {
    #[serde(default)]
    eos_token_id: Option<TokenIds>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl TokenIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(id) => vec![id],
            Self::Many(ids) => ids,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
struct TextRuntimeConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    vocab_size: usize,
    linear_num_value_heads: usize,
    linear_num_key_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: usize,
    head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default)]
    full_attention_interval: usize,
    #[serde(default)]
    mtp_num_hidden_layers: usize,
    #[serde(default)]
    eos_token_id: Option<u32>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Deserialize, Clone)]
struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    partial_rotary_factor: f32,
    #[serde(default)]
    mrope_section: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MropePosition([u32; 3]);

impl MropePosition {
    fn text(position: u32) -> Self {
        Self([position, position, position])
    }

    fn axis_for_frequency(self, frequency: usize, sections: Option<&[usize]>) -> u32 {
        let Some(sections) = sections else {
            return self.0[0];
        };
        if sections.len() != 3 {
            return self.0[0];
        }
        match frequency % 3 {
            1 if frequency / 3 < sections[1] => self.0[1],
            2 if frequency / 3 < sections[2] => self.0[2],
            _ => self.0[0],
        }
    }
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

fn default_partial_rotary_factor() -> f32 {
    0.25
}

fn load_eos_token_ids(path: &Path, configured_eos: u32) -> Result<Vec<u32>, NativeError> {
    let generation_path = path.join("generation_config.json");
    let mut ids = match std::fs::read(&generation_path) {
        Ok(contents) => serde_json::from_slice::<GenerationRuntimeConfig>(&contents)
            .map_err(NativeError::GenerationConfigJson)?
            .eos_token_id
            .map(TokenIds::into_vec)
            .unwrap_or_default(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(source) => {
            return Err(NativeError::ConfigRead {
                path: generation_path,
                source,
            });
        }
    };
    if ids.is_empty() {
        ids.extend([END_OF_MESSAGE_TOKEN_ID, configured_eos]);
    }
    if !ids.contains(&configured_eos) {
        ids.push(configured_eos);
    }
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

impl TextRuntimeConfig {
    fn rope(&self) -> RopeParameters {
        self.rope_parameters.clone().unwrap_or(RopeParameters {
            rope_theta: default_rope_theta(),
            partial_rotary_factor: default_partial_rotary_factor(),
            mrope_section: None,
        })
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id.unwrap_or(DEFAULT_EOS_TOKEN_ID)
    }

    fn max_context(&self) -> usize {
        self.max_position_embeddings
    }
}

/// Read-only native model weights. The field order is intentional: Metal's
/// no-copy buffers release before the safetensors mappings they reference.
pub struct NativeWeights {
    mapped: MappedWeightBuffers,
    store: MlxWeightStore,
}

impl NativeWeights {
    pub fn open(path: impl AsRef<Path>, runtime: &MetalRuntime) -> Result<Self, NativeError> {
        let store = open_mlx_safetensors_dir(path).map_err(NativeError::Model)?;
        let mapped = runtime
            .map_weight_store(&store)
            .map_err(NativeError::Metal)?;
        Ok(Self { mapped, store })
    }

    pub fn q4_matrix(&self, tensor_name: &str) -> Result<Q4AffineMatrix, NativeError> {
        Q4AffineMatrix::from_store(&self.store, tensor_name)
    }

    fn bf16_matrix(&self, tensor_name: &str) -> Result<Bf16Matrix, NativeError> {
        Bf16Matrix::from_store(&self.store, tensor_name)
    }

    pub fn q4_affine_matvec(
        &self,
        runtime: &MetalRuntime,
        matrix: &Q4AffineMatrix,
        input: &[f32],
    ) -> Result<Vec<f32>, NativeError> {
        let mut outputs = self.q4_affine_matvec_batch(runtime, &[matrix], input)?;
        Ok(outputs.remove(0))
    }

    pub fn q4_affine_matvec_batch(
        &self,
        runtime: &MetalRuntime,
        matrices: &[&Q4AffineMatrix],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, NativeError> {
        let jobs = self.mapped_q4_jobs(matrices, input.len())?;
        runtime
            .q4_affine_matvec_mapped_batch(input, &jobs)
            .map_err(NativeError::Metal)
    }

    /// Evaluates a set of Q4 projections for every row in a prompt. Each
    /// returned matrix is row-major and preserves the prompt row order.
    pub fn q4_affine_matmul_batch(
        &self,
        runtime: &MetalRuntime,
        matrices: &[&Q4AffineMatrix],
        input: &[f32],
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, NativeError> {
        if batch_size == 0 || input.is_empty() || input.len() % batch_size != 0 {
            return Err(NativeError::InvalidConfig(
                "a Q4 prefill batch needs non-empty, evenly sized rows".to_owned(),
            ));
        }
        let jobs = self.mapped_q4_jobs(matrices, input.len() / batch_size)?;
        runtime
            .q4_affine_matmul_mapped_batch(input, batch_size, &jobs)
            .map_err(NativeError::Metal)
    }

    /// Computes greedy tokens for a batched Q4 projection without copying
    /// the full output matrix back from Metal.
    fn q4_affine_argmax_batch(
        &self,
        runtime: &MetalRuntime,
        matrix: &Q4AffineMatrix,
        input: &[f32],
        batch_size: usize,
    ) -> Result<Vec<u32>, NativeError> {
        if batch_size == 0 || input.is_empty() || input.len() % batch_size != 0 {
            return Err(NativeError::InvalidConfig(
                "a Q4 argmax batch needs non-empty, evenly sized rows".to_owned(),
            ));
        }
        let jobs = self.mapped_q4_jobs(&[matrix], input.len() / batch_size)?;
        runtime
            .q4_affine_argmax_mapped_batch(input, batch_size, &jobs[0])
            .map_err(NativeError::Metal)
    }

    /// Fuses the MLP's gate/up/down Q4 projections with GPU SwiGLU for all
    /// prompt rows. There is a single GPU completion fence for the chain.
    pub fn q4_affine_mlp_batch(
        &self,
        runtime: &MetalRuntime,
        gate: &Q4AffineMatrix,
        up: &Q4AffineMatrix,
        down: &Q4AffineMatrix,
        input: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, NativeError> {
        if batch_size == 0 || input.is_empty() || input.len() % batch_size != 0 {
            return Err(NativeError::InvalidConfig(
                "an MLP prefill batch needs non-empty, evenly sized rows".to_owned(),
            ));
        }
        let gate_and_up = self.mapped_q4_jobs(&[gate, up], input.len() / batch_size)?;
        let gate_output_elements = usize::try_from(gate.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("MLP gate output rows".to_owned()))?;
        let down_job = self.mapped_q4_jobs(&[down], gate_output_elements)?;
        runtime
            .q4_affine_mlp_mapped_batch(
                input,
                batch_size,
                &gate_and_up[0],
                &gate_and_up[1],
                &down_job[0],
            )
            .map_err(NativeError::Metal)
    }

    fn mapped_q4_jobs<'a>(
        &'a self,
        matrices: &[&Q4AffineMatrix],
        input_elements: usize,
    ) -> Result<Vec<MappedQ4AffineJob<'a>>, NativeError> {
        if matrices.is_empty() {
            return Err(NativeError::InvalidConfig(
                "a Q4 projection batch requires at least one matrix".to_owned(),
            ));
        }
        let mut jobs = Vec::with_capacity(matrices.len());
        for matrix in matrices {
            if input_elements as u64 != matrix.input_elements {
                return Err(NativeError::InputDimension {
                    actual: input_elements,
                    expected: matrix.input_elements,
                });
            }
            let weight_buffer = self
                .mapped
                .buffer(matrix.weight.shard_index)
                .ok_or(NativeError::MissingMappedShard(matrix.weight.shard_index))?;
            let scale_buffer = self
                .mapped
                .buffer(matrix.scales.shard_index)
                .ok_or(NativeError::MissingMappedShard(matrix.scales.shard_index))?;
            let bias_buffer = self
                .mapped
                .buffer(matrix.biases.shard_index)
                .ok_or(NativeError::MissingMappedShard(matrix.biases.shard_index))?;
            let aligned = self
                .mapped
                .offset_is_aligned(matrix.weight.shard_index, matrix.weight.byte_offset, 4)
                .zip(self.mapped.offset_is_aligned(
                    matrix.scales.shard_index,
                    matrix.scales.byte_offset,
                    2,
                ))
                .zip(self.mapped.offset_is_aligned(
                    matrix.biases.shard_index,
                    matrix.biases.byte_offset,
                    2,
                ))
                .map(|((weight, scales), biases)| weight && scales && biases)
                .ok_or(NativeError::MissingMappedShard(matrix.weight.shard_index))?;
            jobs.push(MappedQ4AffineJob::new(
                weight_buffer,
                matrix.weight.byte_offset,
                scale_buffer,
                matrix.scales.byte_offset,
                bias_buffer,
                matrix.biases.byte_offset,
                matrix.output_rows as usize,
                aligned,
            ));
        }
        Ok(jobs)
    }

    pub fn mapped_shard_count(&self) -> usize {
        self.mapped.shard_count()
    }

    fn bf16_gemm(
        &self,
        runtime: &MetalRuntime,
        matrix: &Bf16Matrix,
        input: &[f32],
    ) -> Result<Vec<f32>, NativeError> {
        if input.len() % matrix.input_columns != 0 {
            return Err(NativeError::InputDimension {
                actual: input.len(),
                expected: matrix.input_columns as u64,
            });
        }
        let weight_buffer = self
            .mapped
            .buffer(matrix.weight.shard_index)
            .ok_or(NativeError::MissingMappedShard(matrix.weight.shard_index))?;
        runtime
            .bf16_gemm_mapped(
                input,
                weight_buffer,
                matrix.weight.byte_offset,
                matrix.input_columns,
                matrix.output_rows,
            )
            .map_err(NativeError::Metal)
    }

    pub fn tensor_values_f32(&self, name: &str) -> Result<Vec<f32>, NativeError> {
        self.store
            .tensor_values_f32(name)
            .map_err(NativeError::Model)
    }
}

struct PrefixCacheEntry<S> {
    token_ids: Vec<u32>,
    hidden: Vec<f32>,
    state: S,
}

struct PrefixCache<S> {
    entries: VecDeque<PrefixCacheEntry<S>>,
    total_tokens: usize,
    max_entries: usize,
    max_tokens: usize,
    min_tokens: usize,
}

impl<S> PrefixCache<S> {
    fn from_env() -> Self {
        Self {
            entries: VecDeque::new(),
            total_tokens: 0,
            max_entries: prefix_cache_env(
                "QWEN38_PREFIX_CACHE_MAX_ENTRIES",
                DEFAULT_PREFIX_CACHE_ENTRIES,
            ),
            max_tokens: prefix_cache_env(
                "QWEN38_PREFIX_CACHE_MAX_TOKENS",
                DEFAULT_PREFIX_CACHE_TOKENS,
            ),
            min_tokens: prefix_cache_env(
                "QWEN38_PREFIX_CACHE_MIN_TOKENS",
                DEFAULT_PREFIX_CACHE_MIN_TOKENS,
            ),
        }
    }

    fn find_longest(&self, token_ids: &[u32]) -> Option<usize> {
        longest_prefix_index(
            self.entries
                .iter()
                .enumerate()
                .map(|(index, entry)| (index, entry.token_ids.as_slice())),
            token_ids,
        )
    }

    fn touch(&mut self, index: usize) {
        if index > 0 {
            self.entries.rotate_left(index);
        }
    }

    fn can_store(&self, token_count: usize) -> bool {
        self.max_entries > 0
            && self.max_tokens > 0
            && token_count >= self.min_tokens
            && token_count <= self.max_tokens
    }

    fn insert(&mut self, token_ids: Vec<u32>, hidden: Vec<f32>, state: S) {
        if self.max_entries == 0
            || self.max_tokens == 0
            || token_ids.len() < self.min_tokens
            || token_ids.len() > self.max_tokens
        {
            return;
        }

        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.token_ids == token_ids)
        {
            let removed = self
                .entries
                .remove(index)
                .expect("matching cache entry exists");
            self.total_tokens = self.total_tokens.saturating_sub(removed.token_ids.len());
        }

        while self.entries.len() >= self.max_entries
            || self.total_tokens.saturating_add(token_ids.len()) > self.max_tokens
        {
            let Some(removed) = self.entries.pop_back() else {
                break;
            };
            self.total_tokens = self.total_tokens.saturating_sub(removed.token_ids.len());
        }

        self.total_tokens = self.total_tokens.saturating_add(token_ids.len());
        self.entries.push_front(PrefixCacheEntry {
            token_ids,
            hidden,
            state,
        });
    }
}

fn prefix_cache_env(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => match value.parse::<usize>() {
            Ok(parsed) => parsed,
            Err(_) => {
                eprintln!("warning: {name} must be a non-negative integer; using {default}");
                default
            }
        },
        Err(_) => default,
    }
}

fn longest_prefix_index<'a>(
    entries: impl Iterator<Item = (usize, &'a [u32])>,
    token_ids: &[u32],
) -> Option<usize> {
    entries
        .filter(|(_, prefix)| prefix.len() <= token_ids.len() && token_ids.starts_with(prefix))
        .max_by_key(|(_, prefix)| prefix.len())
        .map(|(index, _)| index)
}

struct PrefixCacheHit {
    token_count: usize,
    hidden: Vec<f32>,
    state: RuntimeState,
}

pub struct NativeEngine {
    descriptor: ModelDescriptor,
    runtime: MetalRuntime,
    weights: NativeWeights,
    tokenizer: Tokenizer,
    model: NativeModel,
    vision: Option<NativeVisionModel>,
    vision_tokens: Option<VisionTokenIds>,
    eos_token_ids: Vec<u32>,
    speculative: SpeculativeDecodeSupport,
    mtp_adapter: Option<MtpAdapter>,
    /// Maximum number of tokens proposed in one MTP round. The default keeps
    /// M4 Pro verification in the more efficient two-row shape; an explicit
    /// environment override is available for device-specific experiments.
    mtp_max_draft_tokens: usize,
    mtp_controller: Mutex<MtpController>,
    prefix_cache: Mutex<PrefixCache<Arc<RuntimeState>>>,
}

#[derive(Debug, Clone, Copy)]
struct VisionTokenIds {
    image_pad: u32,
    vision_start: u32,
    vision_end: u32,
}

impl NativeEngine {
    pub fn open(path: impl AsRef<Path>, model_id: impl Into<String>) -> Result<Self, NativeError> {
        let adapter_path = std::env::var_os("QWEN38_MTP_ADAPTER").map(PathBuf::from);
        Self::open_with_mtp(path, model_id, adapter_path.as_deref())
    }

    pub fn open_with_mtp(
        path: impl AsRef<Path>,
        model_id: impl Into<String>,
        mtp_adapter_path: Option<&Path>,
    ) -> Result<Self, NativeError> {
        let path = path.as_ref();
        let model_id = model_id.into();
        let config_bytes =
            std::fs::read(path.join("config.json")).map_err(|source| NativeError::ConfigRead {
                path: path.join("config.json"),
                source,
            })?;
        let config: RuntimeConfig =
            serde_json::from_slice(&config_bytes).map_err(NativeError::ConfigJson)?;
        let RuntimeConfig {
            text_config,
            vision_config,
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
        } = config;
        let text_config = text_config.ok_or(NativeError::MissingTextConfig)?;
        validate_runtime_config(&text_config)?;

        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|error| NativeError::Tokenizer(error.to_string()))?;
        let runtime = MetalRuntime::new().map_err(NativeError::Metal)?;
        let weights = NativeWeights::open(path, &runtime)?;
        let model = NativeModel::load(&weights, text_config.clone(), &runtime)?;
        let vision = vision_config
            .map(|config| {
                validate_vision_runtime_config(&config)?;
                NativeVisionModel::load(&weights, config)
            })
            .transpose()?;
        let vision_tokens = if vision.is_some() {
            Some(VisionTokenIds {
                image_pad: image_token_id.ok_or_else(|| {
                    NativeError::InvalidConfig("vision_config requires image_token_id".to_owned())
                })?,
                vision_start: vision_start_token_id.ok_or_else(|| {
                    NativeError::InvalidConfig(
                        "vision_config requires vision_start_token_id".to_owned(),
                    )
                })?,
                vision_end: vision_end_token_id.ok_or_else(|| {
                    NativeError::InvalidConfig(
                        "vision_config requires vision_end_token_id".to_owned(),
                    )
                })?,
            })
        } else {
            None
        };
        let context_tokens = u32::try_from(text_config.max_context())
            .map_err(|_| NativeError::DimensionOverflow("max_position_embeddings".to_owned()))?;
        let eos_token_ids = load_eos_token_ids(path, text_config.eos_token_id())?;
        let inspection = inspect_model_dir(path).map_err(NativeError::Preflight)?;
        let mtp_adapter = mtp_adapter_path
            .map(|adapter_path| MtpAdapter::load(adapter_path, &text_config, &runtime))
            .transpose()?;
        let speculative = match &mtp_adapter {
            Some(adapter) => SpeculativeDecodeSupport::from_loaded_adapter(
                &inspection.mtp_support,
                &adapter.support,
                adapter.block_size as u8,
            ),
            None => SpeculativeDecodeSupport::from_mtp_support(&inspection.mtp_support),
        };
        let advertised_draft_tokens = usize::from(speculative.proposal_depth()).max(1);
        let max_draft_tokens = if speculative.is_available() && mtp_adapter.is_some() {
            configured_mtp_draft_limit(advertised_draft_tokens)?
        } else {
            advertised_draft_tokens
        };
        let max_draft_tokens_u8 = u8::try_from(max_draft_tokens).map_err(|_| {
            NativeError::InvalidConfig("MTP draft depth exceeds u8 range".to_owned())
        })?;
        let mtp_controller = MtpController::new(1, max_draft_tokens_u8, max_draft_tokens_u8)
            .map_err(|error| NativeError::InvalidConfig(error.to_string()))?;

        Ok(Self {
            descriptor: ModelDescriptor {
                id: model_id,
                context_tokens,
                execution: ExecutionKind::Native,
            },
            runtime,
            weights,
            tokenizer,
            model,
            vision,
            vision_tokens,
            eos_token_ids,
            speculative,
            mtp_adapter,
            mtp_max_draft_tokens: max_draft_tokens,
            mtp_controller: Mutex::new(mtp_controller),
            prefix_cache: Mutex::new(PrefixCache::from_env()),
        })
    }

    pub fn mapped_shard_count(&self) -> usize {
        self.weights.mapped_shard_count()
    }

    pub fn speculative_decode_support(&self) -> &SpeculativeDecodeSupport {
        &self.speculative
    }

    fn lookup_prefix(&self, token_ids: &[u32]) -> Result<Option<PrefixCacheHit>, NativeError> {
        let (token_count, hidden, state) = {
            let mut cache = self
                .prefix_cache
                .lock()
                .map_err(|_| NativeError::PrefixCachePoisoned)?;
            let Some(index) = cache.find_longest(token_ids) else {
                return Ok(None);
            };
            let entry = cache
                .entries
                .get(index)
                .expect("prefix cache index came from find_longest");
            let token_count = entry.token_ids.len();
            let hidden = entry.hidden.clone();
            let state = Arc::clone(&entry.state);
            cache.touch(index);
            (token_count, hidden, state)
        };
        // The cached state is immutable after insertion. Copy it outside the
        // cache lock so a large Q8 KV prefix does not block other requests.
        let state = state.fork(&self.model, &self.runtime)?;
        Ok(Some(PrefixCacheHit {
            token_count,
            hidden,
            state,
        }))
    }

    fn store_prefix(
        &self,
        token_ids: &[u32],
        hidden: &[f32],
        state: &RuntimeState,
    ) -> Result<(), NativeError> {
        let should_store = {
            let cache = self
                .prefix_cache
                .lock()
                .map_err(|_| NativeError::PrefixCachePoisoned)?;
            cache.can_store(token_ids.len())
        };
        if !should_store {
            return Ok(());
        }
        let state = state.fork(&self.model, &self.runtime)?;
        let mut cache = self
            .prefix_cache
            .lock()
            .map_err(|_| NativeError::PrefixCachePoisoned)?;
        cache.insert(token_ids.to_vec(), hidden.to_vec(), Arc::new(state));
        Ok(())
    }

    fn prepare_images(
        &self,
        request: &GenerationRequest,
    ) -> Result<Vec<PreparedImage>, NativeError> {
        let image_count = request
            .messages
            .iter()
            .map(|message| message.image_count())
            .sum::<usize>();
        if image_count == 0 {
            return Ok(Vec::new());
        }
        let vision = self.vision.as_ref().ok_or_else(|| {
            NativeError::Unavailable(
                "this model does not include a native vision encoder".to_owned(),
            )
        })?;
        request
            .messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|part| match part {
                PromptPart::Image(image) => Some(image),
                PromptPart::Text(_) => None,
            })
            .map(|image| PreparedImage::from_input(image, &vision.config))
            .collect()
    }

    fn render_prompt(
        &self,
        request: &GenerationRequest,
        image_token_counts: &[usize],
    ) -> Result<String, NativeError> {
        if request.messages.is_empty() {
            return Err(NativeError::EmptyPrompt);
        }
        let mut prompt = String::new();
        let mut image_token_counts = image_token_counts.iter().copied();
        let first_is_system = request
            .messages
            .first()
            .is_some_and(|message| message.role == PromptRole::System);
        if !request.tools.is_empty() {
            prompt.push_str("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for tool in &request.tools {
                let tool = json!({
                    "type": "function",
                    "function": {
                        "name": &tool.name,
                        "description": &tool.description,
                        "parameters": &tool.input_schema,
                    }
                });
                let encoded = serde_json::to_string(&tool).map_err(|error| {
                    NativeError::Prompt(format!("cannot encode tool schema: {error}"))
                })?;
                prompt.push('\n');
                prompt.push_str(&encoded);
            }
            prompt.push_str("\n</tools>");
            match &request.tool_choice {
                ToolChoice::Required => {
                    prompt.push_str("\n\nYou must call one of the available functions.")
                }
                ToolChoice::Specific(name) => {
                    prompt.push_str("\n\nYou must call the function `");
                    prompt.push_str(name);
                    prompt.push_str("`.");
                }
                ToolChoice::None | ToolChoice::Auto => {}
            }
            prompt.push_str("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n</IMPORTANT>");
            if first_is_system {
                let content = render_prompt_content(
                    &request.messages[0].content,
                    self.vision_tokens,
                    &mut image_token_counts,
                )?;
                if !content.trim().is_empty() {
                    prompt.push_str("\n\n");
                    prompt.push_str(content.trim());
                }
            }
            prompt.push_str("<|im_end|>\n");
        }

        for (index, message) in request.messages.iter().enumerate() {
            if message.role == PromptRole::System && first_is_system && !request.tools.is_empty() {
                continue;
            }
            match message.role {
                PromptRole::System | PromptRole::User => {
                    let role = if message.role == PromptRole::System {
                        "system"
                    } else {
                        "user"
                    };
                    prompt.push_str("<|im_start|>");
                    prompt.push_str(role);
                    prompt.push('\n');
                    prompt.push_str(&render_prompt_content(
                        &message.content,
                        self.vision_tokens,
                        &mut image_token_counts,
                    )?);
                    prompt.push_str("<|im_end|>\n");
                }
                PromptRole::Assistant => {
                    prompt.push_str("<|im_start|>assistant\n<think>\n");
                    prompt.push_str(
                        message
                            .reasoning_content
                            .as_deref()
                            .unwrap_or_default()
                            .trim(),
                    );
                    prompt.push_str("\n</think>\n\n");
                    prompt.push_str(&render_prompt_content(
                        &message.content,
                        self.vision_tokens,
                        &mut image_token_counts,
                    )?);
                    for call in &message.tool_calls {
                        render_tool_call(&mut prompt, call)?;
                    }
                    prompt.push_str("<|im_end|>\n");
                }
                PromptRole::Tool => {
                    if index == 0 || request.messages[index - 1].role != PromptRole::Tool {
                        prompt.push_str("<|im_start|>user");
                    }
                    prompt.push_str("\n<tool_response>\n");
                    prompt.push_str(&render_prompt_content(
                        &message.content,
                        self.vision_tokens,
                        &mut image_token_counts,
                    )?);
                    prompt.push_str("\n</tool_response>");
                    if index + 1 == request.messages.len()
                        || request.messages[index + 1].role != PromptRole::Tool
                    {
                        prompt.push_str("<|im_end|>\n");
                    }
                }
            }
        }
        prompt.push_str("<|im_start|>assistant\n<think>\n");
        if !request.thinking.enabled {
            prompt.push_str("\n</think>\n\n");
        }
        if image_token_counts.next().is_some() {
            return Err(NativeError::Prompt(
                "the number of rendered image spans does not match image inputs".to_owned(),
            ));
        }
        Ok(prompt)
    }

    fn tokenize(
        &self,
        request: &GenerationRequest,
        image_token_counts: &[usize],
    ) -> Result<Vec<u32>, NativeError> {
        let prompt = self.render_prompt(request, image_token_counts)?;
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|error| NativeError::Tokenizer(error.to_string()))?;
        let ids = encoding.get_ids().to_vec();
        if ids.is_empty() {
            return Err(NativeError::EmptyPrompt);
        }
        Ok(ids)
    }

    fn generate_native(
        &self,
        request: GenerationRequest,
        mut on_event: Option<&mut dyn FnMut(GenerationEvent) -> Result<(), NativeError>>,
    ) -> Result<Generation, NativeError> {
        let images = self.prepare_images(&request)?;
        let image_token_counts: Vec<usize> = images
            .iter()
            .map(PreparedImage::output_token_count)
            .collect();
        let prompt_ids = self.tokenize(&request, &image_token_counts)?;
        let input_tokens = u32::try_from(prompt_ids.len())
            .map_err(|_| NativeError::DimensionOverflow("prompt token count".to_owned()))?;
        let requested = input_tokens
            .checked_add(request.max_tokens)
            .ok_or_else(|| NativeError::DimensionOverflow("requested token count".to_owned()))?;
        if requested > self.descriptor.context_tokens {
            return Err(NativeError::ContextLimit {
                requested,
                maximum: self.descriptor.context_tokens,
            });
        }
        if let Some(callback) = on_event.as_deref_mut() {
            callback(GenerationEvent::Started { input_tokens })?;
        }

        let mut image_features = Vec::new();
        if !images.is_empty() {
            let vision = self.vision.as_ref().ok_or_else(|| {
                NativeError::Unavailable("native vision encoder is unavailable".to_owned())
            })?;
            for image in &images {
                image_features.extend(vision.encode(&self.runtime, &self.weights, image)?);
            }
        }
        let positions = multimodal_positions(&prompt_ids, &images, self.vision_tokens)?;
        if positions.len() != prompt_ids.len() {
            return Err(NativeError::Prompt(
                "the prompt and multimodal position sequences have different lengths".to_owned(),
            ));
        }

        let mut image_feature_index = 0;
        let image_pad = self.vision_tokens.map(|tokens| tokens.image_pad);
        let mut embedding_overrides = Vec::with_capacity(prompt_ids.len());
        for token_id in &prompt_ids {
            if Some(*token_id) == image_pad {
                let feature = image_features.get(image_feature_index).ok_or_else(|| {
                    NativeError::Prompt("image placeholders exceed visual feature count".to_owned())
                })?;
                image_feature_index += 1;
                embedding_overrides.push(Some(feature.as_slice()));
            } else {
                embedding_overrides.push(None);
            }
        }
        if image_feature_index != image_features.len() {
            return Err(NativeError::Prompt(
                "visual features exceed image placeholders in the tokenized prompt".to_owned(),
            ));
        }
        let use_mtp = self.speculative.is_available()
            && self.mtp_adapter.is_some()
            && mtp_request_is_eligible(&request, images.is_empty());
        // Image embeddings are request-local inputs and are intentionally not
        // part of the token-only cache key. MTP also bypasses this cache until
        // the adapter KV state is stored alongside the target state.
        let cacheable = images.is_empty() && !use_mtp;
        let (mut state, mut hidden, mtp_prompt_hidden) = if use_mtp {
            let mut state = RuntimeState::new(&self.model, &self.runtime)?;
            let hidden_rows = self.model.prefill_all(
                &self.runtime,
                &self.weights,
                &mut state,
                &prompt_ids,
                &positions,
                &embedding_overrides,
            )?;
            let final_offset = hidden_rows
                .len()
                .checked_sub(self.model.config.hidden_size)
                .ok_or_else(|| {
                    NativeError::DimensionOverflow("MTP prompt hidden activation".to_owned())
                })?;
            let hidden = hidden_rows[final_offset..].to_vec();
            (state, hidden, Some(hidden_rows))
        } else {
            let prefix_hit = if cacheable {
                self.lookup_prefix(&prompt_ids)?
            } else {
                None
            };
            let (state, hidden) = match prefix_hit {
                Some(hit) if hit.token_count == prompt_ids.len() => (hit.state, hit.hidden),
                Some(hit) => {
                    let token_count = hit.token_count;
                    let mut state = hit.state;
                    let hidden = self.model.prefill(
                        &self.runtime,
                        &self.weights,
                        &mut state,
                        &prompt_ids[token_count..],
                        &positions[token_count..],
                        &embedding_overrides[token_count..],
                    )?;
                    if cacheable {
                        self.store_prefix(&prompt_ids, &hidden, &state)?;
                    }
                    (state, hidden)
                }
                None => {
                    let mut state = RuntimeState::new(&self.model, &self.runtime)?;
                    let hidden = self.model.prefill(
                        &self.runtime,
                        &self.weights,
                        &mut state,
                        &prompt_ids,
                        &positions,
                        &embedding_overrides,
                    )?;
                    if cacheable {
                        self.store_prefix(&prompt_ids, &hidden, &state)?;
                    }
                    (state, hidden)
                }
            };
            (state, hidden, None)
        };

        let mut generated_ids = Vec::new();
        let mut streamed_text = String::new();
        let mut next_logits = self.model.logits(&self.runtime, &self.weights, &hidden)?;
        let mut finish_reason = FinishReason::Length;
        let mut next_position = positions
            .iter()
            .flat_map(|position| position.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        if let Some(prompt_hidden_rows) = mtp_prompt_hidden {
            return self.generate_mtp(
                request,
                input_tokens,
                state,
                hidden,
                next_logits,
                next_position,
                prompt_ids,
                positions,
                prompt_hidden_rows,
                on_event,
            );
        }
        for step in 0..request.max_tokens {
            let token_id = sample_token(
                &next_logits,
                request.temperature,
                request.top_p,
                step as u64 + 1,
            );
            if self.eos_token_ids.contains(&token_id) {
                finish_reason = FinishReason::Stop;
                break;
            }
            generated_ids.push(token_id);
            if let Some(stop_index) =
                token_sequence_stop_index(&self.tokenizer, &generated_ids, &request.stop)?
            {
                generated_ids.truncate(stop_index);
                finish_reason = FinishReason::StopSequence;
                break;
            }
            if let Some(callback) = on_event.as_deref_mut() {
                let decoded = self
                    .tokenizer
                    .decode(&generated_ids, true)
                    .map_err(|error| NativeError::Tokenizer(error.to_string()))?;
                if let Some(delta) = decoded.strip_prefix(&streamed_text) {
                    if !delta.is_empty() {
                        callback(GenerationEvent::RawToken(delta.to_owned()))?;
                    }
                } else if !decoded.is_empty() {
                    // Tokenizers normally append monotonically. If a custom
                    // tokenizer normalizes a prior token, preserve liveness
                    // instead of withholding all remaining output.
                    callback(GenerationEvent::RawToken(decoded.clone()))?;
                }
                streamed_text = decoded;
            }
            hidden = self.model.forward_token(
                &self.runtime,
                &self.weights,
                &mut state,
                token_id,
                MropePosition::text(next_position),
            )?;
            next_position = next_position.saturating_add(1);
            next_logits = self.model.logits(&self.runtime, &self.weights, &hidden)?;
            if step + 1 == request.max_tokens {
                finish_reason = FinishReason::Length;
            }
        }

        let raw = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|error| NativeError::Tokenizer(error.to_string()))?;
        let output_tokens = u32::try_from(generated_ids.len())
            .map_err(|_| NativeError::DimensionOverflow("output token count".to_owned()))?;
        let parts = parse_model_output(&raw, request.thinking, &request.tools)
            .map_err(|error| NativeError::Prompt(error.to_string()))?;
        if !parts.tool_calls.is_empty() {
            finish_reason = FinishReason::ToolCalls;
        }
        Ok(Generation {
            text: parts.text,
            reasoning: parts.reasoning,
            tool_calls: parts.tool_calls,
            input_tokens,
            output_tokens,
            finish_reason,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_mtp(
        &self,
        request: GenerationRequest,
        input_tokens: u32,
        mut state: RuntimeState,
        prompt_hidden: Vec<f32>,
        next_logits: Vec<f32>,
        mut next_position: u32,
        prompt_ids: Vec<u32>,
        prompt_positions: Vec<MropePosition>,
        prompt_hidden_rows: Vec<f32>,
        mut on_event: Option<&mut dyn FnMut(GenerationEvent) -> Result<(), NativeError>>,
    ) -> Result<Generation, NativeError> {
        let adapter = self
            .mtp_adapter
            .as_ref()
            .ok_or_else(|| NativeError::Unavailable("MTP adapter is not loaded".to_owned()))?;
        let mut bonus = argmax(&next_logits);
        let mut adapter_state = adapter.new_request_state(&self.runtime, next_position)?;
        adapter.prefill_prompt(
            &self.runtime,
            &self.weights,
            &self.model,
            &mut adapter_state,
            &prompt_ids,
            &prompt_positions,
            &prompt_hidden_rows,
            bonus,
        )?;
        // NativeModel::prefill returns the trunk hidden before model.norm.
        // Qwen's MTP head applies its own pre_fc_norm_hidden to this value;
        // applying the target final norm here would normalize it twice.
        let mut target_hidden = prompt_hidden;
        let mut generated_ids = Vec::new();
        let mut streamed_text = String::new();
        let mut finish_reason = FinishReason::Length;
        let max_tokens = usize::try_from(request.max_tokens)
            .map_err(|_| NativeError::DimensionOverflow("MTP output token count".to_owned()))?;
        let profile = std::env::var_os("QWEN38_PROFILE").is_some();
        let trace = std::env::var_os("QWEN38_MTP_TRACE").is_some();
        let mtp_started = profile.then(Instant::now);
        let mut rounds = 0_usize;
        let mut drafted_tokens = 0_usize;
        let mut accepted_tokens = 0_usize;
        let mut round_elapsed = Duration::ZERO;
        let mut draft_elapsed = Duration::ZERO;
        let mut verify_elapsed = Duration::ZERO;
        let mut commit_elapsed = Duration::ZERO;

        if !self.emit_mtp_token(
            bonus,
            &request,
            &mut generated_ids,
            &mut streamed_text,
            &mut finish_reason,
            &mut on_event,
        )? {
            log_mtp_profile(
                mtp_started,
                rounds,
                drafted_tokens,
                accepted_tokens,
                round_elapsed,
                draft_elapsed,
                verify_elapsed,
                commit_elapsed,
                generated_ids.len(),
            );
            return self.finish_generation(request, input_tokens, generated_ids, finish_reason);
        }

        while generated_ids.len() < max_tokens {
            let remaining = max_tokens - generated_ids.len();
            if remaining <= 1 {
                if remaining == 1 {
                    // The speculative round needs one target bonus token to
                    // verify a draft. Finish a one-token tail with the normal
                    // target path instead of silently returning short output.
                    let hidden = self.model.forward_token(
                        &self.runtime,
                        &self.weights,
                        &mut state,
                        bonus,
                        MropePosition::text(next_position),
                    )?;
                    let logits = self.model.logits(&self.runtime, &self.weights, &hidden)?;
                    let token_id = argmax(&logits);
                    if self.emit_mtp_token(
                        token_id,
                        &request,
                        &mut generated_ids,
                        &mut streamed_text,
                        &mut finish_reason,
                        &mut on_event,
                    )? && generated_ids.len() == max_tokens
                    {
                        finish_reason = FinishReason::Length;
                    }
                }
                break;
            }
            let controller_depth = self
                .mtp_controller
                .lock()
                .map_err(|_| NativeError::MtpControllerPoisoned)?
                .recommended_depth() as usize;
            let draft_count = controller_depth
                .min(self.mtp_max_draft_tokens)
                .min(remaining.saturating_sub(1));
            if draft_count == 0 {
                break;
            }

            let round_started = profile.then(Instant::now);
            let draft_started = profile.then(Instant::now);
            let draft_tokens = adapter.draft_block(
                &self.runtime,
                &self.weights,
                &self.model,
                &mut adapter_state,
                bonus,
                &target_hidden,
                draft_count,
            )?;
            if let Some(started) = draft_started {
                draft_elapsed += started.elapsed();
            }
            drafted_tokens += draft_tokens.len();
            let mut verify_tokens = Vec::with_capacity(draft_tokens.len() + 1);
            verify_tokens.push(bonus);
            verify_tokens.extend_from_slice(&draft_tokens);
            let mut verify_positions = Vec::with_capacity(verify_tokens.len());
            for offset in 0..verify_tokens.len() {
                verify_positions.push(MropePosition::text(
                    next_position.saturating_add(offset as u32),
                ));
            }
            let verify_overrides = vec![None; verify_tokens.len()];
            let verify_started = profile.then(Instant::now);
            state.begin_speculation(&self.model, &self.runtime)?;
            if std::env::var_os("QWEN38_DISABLE_BATCH_VERIFY").is_none() {
                if let Err(error) = state.prepare_speculation_snapshots(
                    &self.model,
                    &self.runtime,
                    verify_tokens.len().saturating_sub(1),
                ) {
                    let _ = state.rollback_speculation(&self.model, &self.runtime);
                    return Err(error);
                }
            }
            enum VerificationResult {
                Standard(Vec<f32>, Vec<u32>),
                FusedSeed(MetalMtpVerifyResult),
            }

            // The one-draft production mode has no uncommitted adapter KV
            // suffix: its proposal is the seed calculated in the preceding
            // round. That makes the next seed safe to encode immediately
            // after the target verifier chooses its accepted row.
            let use_fused_seed = self.mtp_max_draft_tokens == DEFAULT_MTP_DRAFT_TOKENS
                && draft_count == DEFAULT_MTP_DRAFT_TOKENS
                && adapter_state.round_appended == 0
                && std::env::var_os("QWEN38_DISABLE_BATCH_VERIFY").is_none()
                && std::env::var_os("QWEN38_DISABLE_MTP_FUSED_SEED").is_none();
            let verify_result = (|| {
                if use_fused_seed {
                    self.model
                        .prefill_verify_mtp_seed_gpu(
                            &self.runtime,
                            &self.weights,
                            &mut state,
                            &verify_tokens,
                            &verify_positions,
                            adapter,
                            &mut adapter_state,
                            draft_tokens[0],
                        )
                        .map(VerificationResult::FusedSeed)
                } else if std::env::var_os("QWEN38_DISABLE_BATCH_VERIFY").is_some() {
                    let verify_hidden = self.model.prefill_all(
                        &self.runtime,
                        &self.weights,
                        &mut state,
                        &verify_tokens,
                        &verify_positions,
                        &verify_overrides,
                    )?;
                    let target_tokens = self.model.argmax_logits_rows(
                        &self.runtime,
                        &self.weights,
                        &verify_hidden,
                        verify_tokens.len(),
                    )?;
                    Ok::<_, NativeError>(VerificationResult::Standard(verify_hidden, target_tokens))
                } else {
                    self.model
                        .prefill_verify_gpu(
                            &self.runtime,
                            &self.weights,
                            &mut state,
                            &verify_tokens,
                            &verify_positions,
                        )
                        .map(|(hidden, tokens)| VerificationResult::Standard(hidden, tokens))
                }
            })();
            let (verify_hidden, accepted, target_bonus, fused_seed, target_tokens) =
                match verify_result {
                    Ok(VerificationResult::FusedSeed(result)) => {
                        (Vec::new(), result.accepted, result.target_bonus, true, None)
                    }
                    Ok(VerificationResult::Standard(hidden, tokens)) => {
                        let accepted = accepted_token_count(&draft_tokens, &tokens, draft_count);
                        let target_bonus = match tokens.get(accepted).copied() {
                            Some(token) => token,
                            None => {
                                let _ = state.rollback_speculation(&self.model, &self.runtime);
                                return Err(NativeError::VectorLengthMismatch {
                                    actual: tokens.len(),
                                    expected: draft_count + 1,
                                });
                            }
                        };
                        (hidden, accepted, target_bonus, false, Some(tokens))
                    }
                    Err(error) => {
                        let _ = state.rollback_speculation(&self.model, &self.runtime);
                        return Err(error);
                    }
                };
            if let Some(started) = verify_started {
                verify_elapsed += started.elapsed();
            }
            if trace {
                if let Some(target_tokens) = target_tokens {
                    eprintln!(
                        "mtp round={} position={} bonus={} draft={:?} target={:?} accepted={} target_bonus={}",
                        rounds,
                        next_position,
                        bonus,
                        draft_tokens,
                        target_tokens,
                        accepted,
                        target_bonus,
                    );
                } else {
                    eprintln!(
                        "mtp round={} position={} bonus={} draft={:?} accepted={} target_bonus={} fused_seed=true",
                        rounds,
                        next_position,
                        bonus,
                        draft_tokens,
                        accepted,
                        target_bonus,
                    );
                }
            }

            let commit_started = profile.then(Instant::now);
            if accepted == draft_count {
                // A fully accepted block already has the exact target state
                // needed by the next round. Swap the shadow DeltaNet buffers
                // into the request without copying recurrent state.
                if let Err(error) = state.commit_speculation(&self.model) {
                    let _ = state.rollback_speculation(&self.model, &self.runtime);
                    return Err(error);
                }
            } else if std::env::var_os("QWEN38_DISABLE_BATCH_VERIFY").is_none()
                && state.has_speculation_snapshots(&self.model)
            {
                // The verifier already produced a state image after every
                // causal row. Restore the accepted row and retain exactly its
                // KV suffix instead of re-executing the target prefix.
                if let Err(error) =
                    state.commit_speculation_prefix(&self.model, &self.runtime, accepted + 1)
                {
                    let _ = state.rollback_speculation(&self.model, &self.runtime);
                    return Err(error);
                }
            } else {
                // Full-attention KV bytes were appended in place, while the
                // DeltaNet result lives only in shadow buffers. Discard the
                // transaction before replaying the accepted target prefix.
                state.rollback_speculation(&self.model, &self.runtime)?;
                let mut committed_tokens = Vec::with_capacity(accepted + 1);
                committed_tokens.push(bonus);
                committed_tokens.extend_from_slice(&draft_tokens[..accepted]);
                let committed_positions = (0..committed_tokens.len())
                    .map(|offset| MropePosition::text(next_position.saturating_add(offset as u32)))
                    .collect::<Vec<_>>();
                for (token_id, position) in committed_tokens
                    .iter()
                    .copied()
                    .zip(committed_positions.iter().copied())
                {
                    let _ = self.model.forward_token(
                        &self.runtime,
                        &self.weights,
                        &mut state,
                        token_id,
                        position,
                    )?;
                }
            }

            next_position = next_position.saturating_add(accepted as u32 + 1);
            if !fused_seed {
                let hidden_offset = accepted
                    .checked_mul(self.model.config.hidden_size)
                    .ok_or_else(|| {
                        NativeError::DimensionOverflow("MTP target hidden offset".to_owned())
                    })?;
                let hidden_end = hidden_offset
                    .checked_add(self.model.config.hidden_size)
                    .ok_or_else(|| {
                        NativeError::DimensionOverflow("MTP target hidden end".to_owned())
                    })?;
                target_hidden = verify_hidden
                    .get(hidden_offset..hidden_end)
                    .ok_or_else(|| NativeError::VectorLengthMismatch {
                        actual: verify_hidden.len(),
                        expected: (accepted + 1) * self.model.config.hidden_size,
                    })?
                    .to_vec();
            }

            let mut stopped = false;
            for token_id in draft_tokens
                .iter()
                .copied()
                .take(accepted)
                .chain(std::iter::once(target_bonus))
            {
                if !self.emit_mtp_token(
                    token_id,
                    &request,
                    &mut generated_ids,
                    &mut streamed_text,
                    &mut finish_reason,
                    &mut on_event,
                )? || generated_ids.len() >= max_tokens
                {
                    stopped = true;
                    break;
                }
            }
            if stopped {
                if let Some(started) = commit_started {
                    commit_elapsed += started.elapsed();
                }
                rounds += 1;
                accepted_tokens += accepted;
                if let Some(started) = round_started {
                    round_elapsed += started.elapsed();
                }
                break;
            }

            if !fused_seed {
                adapter.accept_verified(
                    &self.runtime,
                    &self.weights,
                    &self.model,
                    &mut adapter_state,
                    &draft_tokens,
                    accepted,
                    target_bonus,
                    &verify_hidden,
                )?;
            }
            if let Some(started) = commit_started {
                commit_elapsed += started.elapsed();
            }
            self.mtp_controller
                .lock()
                .map_err(|_| NativeError::MtpControllerPoisoned)?
                .observe(draft_count as u8, accepted as u8)
                .map_err(|error| NativeError::InvalidConfig(error.to_string()))?;
            bonus = target_bonus;
            rounds += 1;
            accepted_tokens += accepted;
            if let Some(started) = round_started {
                round_elapsed += started.elapsed();
            }
        }

        if generated_ids.len() == max_tokens {
            finish_reason = FinishReason::Length;
        }
        log_mtp_profile(
            mtp_started,
            rounds,
            drafted_tokens,
            accepted_tokens,
            round_elapsed,
            draft_elapsed,
            verify_elapsed,
            commit_elapsed,
            generated_ids.len(),
        );
        self.finish_generation(request, input_tokens, generated_ids, finish_reason)
    }

    fn emit_mtp_token(
        &self,
        token_id: u32,
        request: &GenerationRequest,
        generated_ids: &mut Vec<u32>,
        streamed_text: &mut String,
        finish_reason: &mut FinishReason,
        on_event: &mut Option<&mut dyn FnMut(GenerationEvent) -> Result<(), NativeError>>,
    ) -> Result<bool, NativeError> {
        if self.eos_token_ids.contains(&token_id) {
            *finish_reason = FinishReason::Stop;
            return Ok(false);
        }
        generated_ids.push(token_id);
        if let Some(stop_index) =
            token_sequence_stop_index(&self.tokenizer, generated_ids, &request.stop)?
        {
            generated_ids.truncate(stop_index);
            *finish_reason = FinishReason::StopSequence;
            return Ok(false);
        }
        if let Some(callback) = on_event.as_deref_mut() {
            let decoded = self
                .tokenizer
                .decode(generated_ids, true)
                .map_err(|error| NativeError::Tokenizer(error.to_string()))?;
            if let Some(delta) = decoded.strip_prefix(streamed_text.as_str()) {
                if !delta.is_empty() {
                    callback(GenerationEvent::RawToken(delta.to_owned()))?;
                }
            } else if !decoded.is_empty() {
                callback(GenerationEvent::RawToken(decoded.clone()))?;
            }
            *streamed_text = decoded;
        }
        Ok(true)
    }

    fn finish_generation(
        &self,
        request: GenerationRequest,
        input_tokens: u32,
        generated_ids: Vec<u32>,
        finish_reason: FinishReason,
    ) -> Result<Generation, NativeError> {
        let raw = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|error| NativeError::Tokenizer(error.to_string()))?;
        let output_tokens = u32::try_from(generated_ids.len())
            .map_err(|_| NativeError::DimensionOverflow("output token count".to_owned()))?;
        let parts = parse_model_output(&raw, request.thinking, &request.tools)
            .map_err(|error| NativeError::Prompt(error.to_string()))?;
        let finish_reason = if parts.tool_calls.is_empty() {
            finish_reason
        } else {
            FinishReason::ToolCalls
        };
        Ok(Generation {
            text: parts.text,
            reasoning: parts.reasoning,
            tool_calls: parts.tool_calls,
            input_tokens,
            output_tokens,
            finish_reason,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn log_mtp_profile(
    started: Option<Instant>,
    rounds: usize,
    drafted_tokens: usize,
    accepted_tokens: usize,
    round_elapsed: Duration,
    draft_elapsed: Duration,
    verify_elapsed: Duration,
    commit_elapsed: Duration,
    output_tokens: usize,
) {
    let Some(started) = started else {
        return;
    };
    let elapsed = started.elapsed();
    let acceptance = if drafted_tokens == 0 {
        0.0
    } else {
        accepted_tokens as f64 / drafted_tokens as f64
    };
    let token_per_second = if elapsed.is_zero() {
        0.0
    } else {
        output_tokens as f64 / elapsed.as_secs_f64()
    };
    eprintln!(
        "mtp rounds={rounds} drafted={drafted_tokens} accepted={accepted_tokens} acceptance={acceptance:.3} round_ms={:.3} draft_ms={:.3} verify_ms={:.3} commit_ms={:.3} output_tokens={output_tokens} decode_tok_per_s={token_per_second:.3}",
        round_elapsed.as_secs_f64() * 1_000.0,
        draft_elapsed.as_secs_f64() * 1_000.0,
        verify_elapsed.as_secs_f64() * 1_000.0,
        commit_elapsed.as_secs_f64() * 1_000.0,
    );
}

fn render_prompt_content(
    content: &[PromptPart],
    vision_tokens: Option<VisionTokenIds>,
    image_token_counts: &mut impl Iterator<Item = usize>,
) -> Result<String, NativeError> {
    let mut rendered = String::new();
    for part in content {
        match part {
            PromptPart::Text(text) => rendered.push_str(text),
            PromptPart::Image(_) => {
                vision_tokens.ok_or_else(|| {
                    NativeError::Unavailable("this model does not support image inputs".to_owned())
                })?;
                let count = image_token_counts.next().ok_or_else(|| {
                    NativeError::Prompt(
                        "image token count is missing for an input image".to_owned(),
                    )
                })?;
                rendered.push_str("<|vision_start|>");
                for _ in 0..count {
                    rendered.push_str("<|image_pad|>");
                }
                rendered.push_str("<|vision_end|>");
            }
        }
    }
    Ok(rendered)
}

fn render_tool_call(prompt: &mut String, call: &crate::api::ToolCall) -> Result<(), NativeError> {
    prompt.push_str("\n<tool_call>\n<function=");
    prompt.push_str(&call.name);
    prompt.push_str(">\n");
    let arguments = call.arguments.as_object().ok_or_else(|| {
        NativeError::Prompt("tool call arguments must be a JSON object".to_owned())
    })?;
    for (name, value) in arguments {
        prompt.push_str("<parameter=");
        prompt.push_str(name);
        prompt.push_str(">\n");
        match value {
            serde_json::Value::String(value) => prompt.push_str(value),
            value => {
                let encoded = serde_json::to_string(value).map_err(|error| {
                    NativeError::Prompt(format!("cannot encode tool argument: {error}"))
                })?;
                prompt.push_str(&encoded);
            }
        }
        prompt.push_str("\n</parameter>\n");
    }
    prompt.push_str("</function>\n</tool_call>");
    Ok(())
}

impl InferenceEngine for NativeEngine {
    fn descriptor(&self) -> ModelDescriptor {
        self.descriptor.clone()
    }

    fn estimate_prompt_tokens(&self, request: &GenerationRequest) -> Result<u32, EngineError> {
        let images = self.prepare_images(request).map_err(native_engine_error)?;
        let image_token_counts: Vec<usize> = images
            .iter()
            .map(PreparedImage::output_token_count)
            .collect();
        self.tokenize(request, &image_token_counts)
            .and_then(|ids| {
                u32::try_from(ids.len())
                    .map_err(|_| NativeError::DimensionOverflow("prompt token count".to_owned()))
            })
            .map_err(native_engine_error)
    }

    fn generate(&self, request: GenerationRequest) -> Result<Generation, EngineError> {
        self.generate_native(request, None)
            .map_err(native_engine_error)
    }

    fn generate_stream(
        &self,
        request: GenerationRequest,
        callback: &mut dyn FnMut(GenerationEvent) -> Result<(), EngineError>,
    ) -> Result<Generation, EngineError> {
        let mut forward_event =
            |event| callback(event).map_err(|error| NativeError::Streaming(error.to_string()));
        let generation = self
            .generate_native(request, Some(&mut forward_event))
            .map_err(native_engine_error)?;
        callback(GenerationEvent::Finished(generation.clone()))?;
        Ok(generation)
    }
}

fn native_engine_error(error: NativeError) -> EngineError {
    match error {
        NativeError::ContextLimit { requested, maximum } => {
            EngineError::ContextLimit { requested, maximum }
        }
        NativeError::Unavailable(message) => EngineError::Unavailable(message),
        NativeError::Image(message) => EngineError::InvalidRequest(message),
        other => EngineError::Failure(other.to_string()),
    }
}

struct NativeVisionModel {
    config: VisionRuntimeConfig,
    patch_projection: Bf16Matrix,
    patch_bias: Vec<f32>,
    position_embeddings: Vec<f32>,
    blocks: Vec<VisionBlockWeights>,
    merger: VisionMergerWeights,
}

struct VisionBlockWeights {
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    qkv: Bf16Matrix,
    qkv_bias: Vec<f32>,
    projection: Bf16Matrix,
    projection_bias: Vec<f32>,
    norm2_weight: Vec<f32>,
    norm2_bias: Vec<f32>,
    mlp_in: Bf16Matrix,
    mlp_in_bias: Vec<f32>,
    mlp_out: Bf16Matrix,
    mlp_out_bias: Vec<f32>,
}

struct VisionMergerWeights {
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    linear_fc1: Bf16Matrix,
    linear_fc1_bias: Vec<f32>,
    linear_fc2: Bf16Matrix,
    linear_fc2_bias: Vec<f32>,
}

impl NativeVisionModel {
    fn load(weights: &NativeWeights, config: VisionRuntimeConfig) -> Result<Self, NativeError> {
        let patch_projection = weights.bf16_matrix("vision_tower.patch_embed.proj.weight")?;
        let expected_patch_columns = config
            .in_channels
            .checked_mul(config.temporal_patch_size)
            .and_then(|value| value.checked_mul(config.patch_size))
            .and_then(|value| value.checked_mul(config.patch_size))
            .ok_or_else(|| NativeError::DimensionOverflow("vision patch dimensions".to_owned()))?;
        if patch_projection.input_columns != expected_patch_columns
            || patch_projection.output_rows != config.hidden_size
        {
            return Err(NativeError::InvalidConfig(
                "vision patch projection dimensions do not match vision_config".to_owned(),
            ));
        }
        let patch_bias = load_vector(
            weights,
            "vision_tower.patch_embed.proj.bias",
            config.hidden_size,
        )?;
        let position_embeddings = weights.tensor_values_f32("vision_tower.pos_embed.weight")?;
        let expected_positions = config
            .num_position_embeddings
            .checked_mul(config.hidden_size)
            .ok_or_else(|| {
                NativeError::DimensionOverflow("vision position embeddings".to_owned())
            })?;
        if position_embeddings.len() != expected_positions {
            return Err(NativeError::WrongVectorLength {
                name: "vision_tower.pos_embed.weight".to_owned(),
                actual: position_embeddings.len(),
                expected: expected_positions,
            });
        }

        let mut blocks = Vec::with_capacity(config.depth);
        for index in 0..config.depth {
            let prefix = format!("vision_tower.blocks.{index}");
            let qkv = weights.bf16_matrix(&format!("{prefix}.attn.qkv.weight"))?;
            let projection = weights.bf16_matrix(&format!("{prefix}.attn.proj.weight"))?;
            let mlp_in = weights.bf16_matrix(&format!("{prefix}.mlp.linear_fc1.weight"))?;
            let mlp_out = weights.bf16_matrix(&format!("{prefix}.mlp.linear_fc2.weight"))?;
            if qkv.input_columns != config.hidden_size
                || qkv.output_rows != config.hidden_size * 3
                || projection.input_columns != config.hidden_size
                || projection.output_rows != config.hidden_size
                || mlp_in.input_columns != config.hidden_size
                || mlp_in.output_rows != config.intermediate_size
                || mlp_out.input_columns != config.intermediate_size
                || mlp_out.output_rows != config.hidden_size
            {
                return Err(NativeError::InvalidConfig(format!(
                    "vision block {index} has dimensions incompatible with vision_config"
                )));
            }
            blocks.push(VisionBlockWeights {
                norm1_weight: load_vector(
                    weights,
                    &format!("{prefix}.norm1.weight"),
                    config.hidden_size,
                )?,
                norm1_bias: load_vector(
                    weights,
                    &format!("{prefix}.norm1.bias"),
                    config.hidden_size,
                )?,
                qkv,
                qkv_bias: load_vector(
                    weights,
                    &format!("{prefix}.attn.qkv.bias"),
                    config.hidden_size * 3,
                )?,
                projection,
                projection_bias: load_vector(
                    weights,
                    &format!("{prefix}.attn.proj.bias"),
                    config.hidden_size,
                )?,
                norm2_weight: load_vector(
                    weights,
                    &format!("{prefix}.norm2.weight"),
                    config.hidden_size,
                )?,
                norm2_bias: load_vector(
                    weights,
                    &format!("{prefix}.norm2.bias"),
                    config.hidden_size,
                )?,
                mlp_in,
                mlp_in_bias: load_vector(
                    weights,
                    &format!("{prefix}.mlp.linear_fc1.bias"),
                    config.intermediate_size,
                )?,
                mlp_out,
                mlp_out_bias: load_vector(
                    weights,
                    &format!("{prefix}.mlp.linear_fc2.bias"),
                    config.hidden_size,
                )?,
            });
        }
        let merger = VisionMergerWeights {
            norm_weight: load_vector(
                weights,
                "vision_tower.merger.norm.weight",
                config.hidden_size,
            )?,
            norm_bias: load_vector(weights, "vision_tower.merger.norm.bias", config.hidden_size)?,
            linear_fc1: weights.bf16_matrix("vision_tower.merger.linear_fc1.weight")?,
            linear_fc1_bias: load_vector(
                weights,
                "vision_tower.merger.linear_fc1.bias",
                config.hidden_size * config.spatial_merge_size * config.spatial_merge_size,
            )?,
            linear_fc2: weights.bf16_matrix("vision_tower.merger.linear_fc2.weight")?,
            linear_fc2_bias: load_vector(
                weights,
                "vision_tower.merger.linear_fc2.bias",
                config.out_hidden_size,
            )?,
        };
        let merged_size = config
            .hidden_size
            .checked_mul(config.spatial_merge_size)
            .and_then(|value| value.checked_mul(config.spatial_merge_size))
            .ok_or_else(|| {
                NativeError::DimensionOverflow("vision merged hidden size".to_owned())
            })?;
        if merger.linear_fc1.input_columns != merged_size
            || merger.linear_fc1.output_rows != merged_size
            || merger.linear_fc2.input_columns != merged_size
            || merger.linear_fc2.output_rows != config.out_hidden_size
        {
            return Err(NativeError::InvalidConfig(
                "vision merger dimensions do not match vision_config".to_owned(),
            ));
        }
        Ok(Self {
            config,
            patch_projection,
            patch_bias,
            position_embeddings,
            blocks,
            merger,
        })
    }

    fn encode(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        image: &PreparedImage,
    ) -> Result<Vec<Vec<f32>>, NativeError> {
        let mut hidden = weights.bf16_gemm(runtime, &self.patch_projection, &image.patches)?;
        add_bias_rows(&mut hidden, &self.patch_bias)?;
        add_in_place(
            &mut hidden,
            &interpolated_position_embeddings(
                &self.position_embeddings,
                self.config.hidden_size,
                self.config.num_position_embeddings,
                image.grid_height,
                image.grid_width,
                self.config.spatial_merge_size,
            )?,
        )?;
        for block in &self.blocks {
            let normalized =
                layer_norm_rows(&hidden, &block.norm1_weight, &block.norm1_bias, 1e-6)?;
            let mut qkv = weights.bf16_gemm(runtime, &block.qkv, &normalized)?;
            add_bias_rows(&mut qkv, &block.qkv_bias)?;
            let (queries, keys, values) = split_vision_qkv(
                &qkv,
                self.config.hidden_size,
                self.config.num_heads,
                image.grid_height,
                image.grid_width,
                self.config.spatial_merge_size,
            )?;
            let attention = runtime
                .vision_attention(
                    &queries,
                    &keys,
                    &values,
                    image.patch_count(),
                    self.config.num_heads,
                    self.config.hidden_size / self.config.num_heads,
                )
                .map_err(NativeError::Metal)?;
            let mut projected = weights.bf16_gemm(runtime, &block.projection, &attention)?;
            add_bias_rows(&mut projected, &block.projection_bias)?;
            add_in_place(&mut hidden, &projected)?;
            let normalized =
                layer_norm_rows(&hidden, &block.norm2_weight, &block.norm2_bias, 1e-6)?;
            let mut intermediate = weights.bf16_gemm(runtime, &block.mlp_in, &normalized)?;
            add_bias_rows(&mut intermediate, &block.mlp_in_bias)?;
            gelu_tanh_in_place(&mut intermediate);
            let mut mlp = weights.bf16_gemm(runtime, &block.mlp_out, &intermediate)?;
            add_bias_rows(&mut mlp, &block.mlp_out_bias)?;
            add_in_place(&mut hidden, &mlp)?;
        }

        let normalized = layer_norm_rows(
            &hidden,
            &self.merger.norm_weight,
            &self.merger.norm_bias,
            1e-6,
        )?;
        let merged = merge_vision_patches(
            &normalized,
            image.grid_height,
            image.grid_width,
            self.config.hidden_size,
            self.config.spatial_merge_size,
        )?;
        let mut projected = weights.bf16_gemm(runtime, &self.merger.linear_fc1, &merged)?;
        add_bias_rows(&mut projected, &self.merger.linear_fc1_bias)?;
        gelu_tanh_in_place(&mut projected);
        let mut output = weights.bf16_gemm(runtime, &self.merger.linear_fc2, &projected)?;
        add_bias_rows(&mut output, &self.merger.linear_fc2_bias)?;
        rows_from_flat(output, self.config.out_hidden_size)
    }
}

struct PreparedImage {
    patches: Vec<f32>,
    grid_height: usize,
    grid_width: usize,
    merge_size: usize,
}

impl PreparedImage {
    fn from_input(input: &InputImage, config: &VisionRuntimeConfig) -> Result<Self, NativeError> {
        let decoded = image::load_from_memory(&input.bytes)
            .map_err(|error| NativeError::Image(format!("cannot decode image: {error}")))?
            .to_rgb8();
        let (width, height) = decoded.dimensions();
        if width == 0 || height == 0 {
            return Err(NativeError::Image(
                "image dimensions must be non-zero".to_owned(),
            ));
        }
        let factor = config
            .patch_size
            .checked_mul(config.spatial_merge_size)
            .ok_or_else(|| NativeError::DimensionOverflow("image resize factor".to_owned()))?;
        let (target_height, target_width) =
            smart_resize(height as usize, width as usize, factor, 65_536, 262_144)?;
        let resized = image::imageops::resize(
            &decoded,
            u32::try_from(target_width)
                .map_err(|_| NativeError::DimensionOverflow("image width".to_owned()))?,
            u32::try_from(target_height)
                .map_err(|_| NativeError::DimensionOverflow("image height".to_owned()))?,
            FilterType::CatmullRom,
        );
        let grid_height = target_height / config.patch_size;
        let grid_width = target_width / config.patch_size;
        let patch_elements = config
            .in_channels
            .checked_mul(config.temporal_patch_size)
            .and_then(|value| value.checked_mul(config.patch_size))
            .and_then(|value| value.checked_mul(config.patch_size))
            .ok_or_else(|| NativeError::DimensionOverflow("image patch elements".to_owned()))?;
        let patch_count = grid_height
            .checked_mul(grid_width)
            .ok_or_else(|| NativeError::DimensionOverflow("image patch count".to_owned()))?;
        let mut patches = Vec::with_capacity(
            patch_count
                .checked_mul(patch_elements)
                .ok_or_else(|| NativeError::DimensionOverflow("image patch bytes".to_owned()))?,
        );
        // The Qwen processor emits patches by temporal group, merge block,
        // then pixels in row-major order. A static image is duplicated across
        // the temporal dimension before this transform.
        for block_y in 0..grid_height / config.spatial_merge_size {
            for block_x in 0..grid_width / config.spatial_merge_size {
                for intra_y in 0..config.spatial_merge_size {
                    for intra_x in 0..config.spatial_merge_size {
                        let patch_y =
                            (block_y * config.spatial_merge_size + intra_y) * config.patch_size;
                        let patch_x =
                            (block_x * config.spatial_merge_size + intra_x) * config.patch_size;
                        for channel in 0..config.in_channels {
                            for _temporal in 0..config.temporal_patch_size {
                                for y in 0..config.patch_size {
                                    for x in 0..config.patch_size {
                                        let pixel = resized.get_pixel(
                                            u32::try_from(patch_x + x).map_err(|_| {
                                                NativeError::DimensionOverflow("image x".to_owned())
                                            })?,
                                            u32::try_from(patch_y + y).map_err(|_| {
                                                NativeError::DimensionOverflow("image y".to_owned())
                                            })?,
                                        );
                                        patches.push((pixel[channel] as f32 / 255.0 - 0.5) / 0.5);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if patches.len() != patch_count * patch_elements {
            return Err(NativeError::Image(
                "image preprocessing produced an unexpected patch layout".to_owned(),
            ));
        }
        Ok(Self {
            patches,
            grid_height,
            grid_width,
            merge_size: config.spatial_merge_size,
        })
    }

    fn patch_count(&self) -> usize {
        self.grid_height * self.grid_width
    }

    fn output_token_count(&self) -> usize {
        self.patch_count() / (self.merge_size * self.merge_size)
    }
}

fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), NativeError> {
    if factor == 0 {
        return Err(NativeError::Image(format!(
            "image resize factor {factor} must be greater than zero"
        )));
    }
    if height.max(width) > height.min(width).saturating_mul(200) {
        return Err(NativeError::Image(
            "image aspect ratio must not exceed 200:1".to_owned(),
        ));
    }
    let round_to_factor =
        |value: usize| -> usize { ((value + factor / 2) / factor).max(1) * factor };
    let mut target_height = round_to_factor(height);
    let mut target_width = round_to_factor(width);
    let pixels = target_height
        .checked_mul(target_width)
        .ok_or_else(|| NativeError::DimensionOverflow("image pixel count".to_owned()))?;
    if pixels > max_pixels {
        let scale = ((height * width) as f64 / max_pixels as f64).sqrt();
        target_height = ((height as f64 / scale / factor as f64).floor() as usize).max(1) * factor;
        target_width = ((width as f64 / scale / factor as f64).floor() as usize).max(1) * factor;
    } else if pixels < min_pixels {
        let scale = (min_pixels as f64 / (height * width) as f64).sqrt();
        target_height = ((height as f64 * scale / factor as f64).ceil() as usize).max(1) * factor;
        target_width = ((width as f64 * scale / factor as f64).ceil() as usize).max(1) * factor;
    }
    Ok((target_height, target_width))
}

fn multimodal_positions(
    prompt_ids: &[u32],
    images: &[PreparedImage],
    tokens: Option<VisionTokenIds>,
) -> Result<Vec<MropePosition>, NativeError> {
    if images.is_empty() {
        return Ok((0..prompt_ids.len())
            .map(|position| MropePosition::text(position as u32))
            .collect());
    }
    let tokens = tokens.ok_or_else(|| {
        NativeError::Unavailable("this model does not include multimodal token ids".to_owned())
    })?;
    let mut positions = Vec::with_capacity(prompt_ids.len());
    let mut image_index = 0;
    let mut offset = 0_u32;
    let mut cursor = 0;
    while cursor < prompt_ids.len() {
        if prompt_ids[cursor] != tokens.vision_start {
            positions.push(MropePosition::text(offset));
            offset = offset.saturating_add(1);
            cursor += 1;
            continue;
        }
        let image = images.get(image_index).ok_or_else(|| {
            NativeError::Prompt("vision start token has no matching input image".to_owned())
        })?;
        positions.push(MropePosition::text(offset));
        cursor += 1;
        let expected = image.output_token_count();
        let merged_width = image.grid_width / image.merge_size;
        let merged_height = image.grid_height / image.merge_size;
        for patch in 0..expected {
            if prompt_ids.get(cursor).copied() != Some(tokens.image_pad) {
                return Err(NativeError::Prompt(
                    "image token expansion does not match the rendered visual span".to_owned(),
                ));
            }
            let temporal = 0_u32;
            let row = u32::try_from(patch / merged_width)
                .map_err(|_| NativeError::DimensionOverflow("visual row position".to_owned()))?;
            let column = u32::try_from(patch % merged_width)
                .map_err(|_| NativeError::DimensionOverflow("visual column position".to_owned()))?;
            if row as usize >= merged_height {
                return Err(NativeError::Prompt(
                    "visual grid position overflows image height".to_owned(),
                ));
            }
            positions.push(MropePosition([
                offset.saturating_add(1).saturating_add(temporal),
                offset.saturating_add(1).saturating_add(row),
                offset.saturating_add(1).saturating_add(column),
            ]));
            cursor += 1;
        }
        if prompt_ids.get(cursor).copied() != Some(tokens.vision_end) {
            return Err(NativeError::Prompt(
                "visual span is missing its vision end token".to_owned(),
            ));
        }
        let next_text_position = offset
            .saturating_add(1)
            .saturating_add(u32::try_from(merged_height.max(merged_width)).map_err(|_| {
                NativeError::DimensionOverflow("visual position offset".to_owned())
            })?);
        positions.push(MropePosition::text(next_text_position));
        offset = next_text_position.saturating_add(1);
        cursor += 1;
        image_index += 1;
    }
    if image_index != images.len() {
        return Err(NativeError::Prompt(
            "input images do not all appear in the rendered prompt".to_owned(),
        ));
    }
    Ok(positions)
}

fn add_bias_rows(values: &mut [f32], bias: &[f32]) -> Result<(), NativeError> {
    if bias.is_empty() || values.len() % bias.len() != 0 {
        return Err(NativeError::VectorLengthMismatch {
            actual: values.len(),
            expected: bias.len(),
        });
    }
    for row in values.chunks_exact_mut(bias.len()) {
        for (value, bias) in row.iter_mut().zip(bias) {
            *value += *bias;
        }
    }
    Ok(())
}

fn layer_norm_rows(
    values: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<Vec<f32>, NativeError> {
    if weight.is_empty() || weight.len() != bias.len() || values.len() % weight.len() != 0 {
        return Err(NativeError::VectorLengthMismatch {
            actual: values.len(),
            expected: weight.len(),
        });
    }
    let mut result = Vec::with_capacity(values.len());
    for row in values.chunks_exact(weight.len()) {
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        let variance = row
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / row.len() as f32;
        let scale = (variance + eps).sqrt().recip();
        result.extend(
            row.iter()
                .zip(weight)
                .zip(bias)
                .map(|((value, weight), bias)| (*value - mean) * scale * weight + bias),
        );
    }
    Ok(result)
}

fn gelu_tanh_in_place(values: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    for value in values {
        let original = *value;
        *value = 0.5
            * original
            * (1.0 + (SQRT_2_OVER_PI * (original + 0.044_715 * original.powi(3))).tanh());
    }
}

fn rows_from_flat(values: Vec<f32>, width: usize) -> Result<Vec<Vec<f32>>, NativeError> {
    if width == 0 || values.len() % width != 0 {
        return Err(NativeError::VectorLengthMismatch {
            actual: values.len(),
            expected: width,
        });
    }
    Ok(values.chunks_exact(width).map(|row| row.to_vec()).collect())
}

fn interpolated_position_embeddings(
    table: &[f32],
    hidden_size: usize,
    position_count: usize,
    grid_height: usize,
    grid_width: usize,
    merge_size: usize,
) -> Result<Vec<f32>, NativeError> {
    let source_side = (position_count as f64).sqrt() as usize;
    if source_side * source_side != position_count || table.len() != position_count * hidden_size {
        return Err(NativeError::InvalidConfig(
            "vision positional embedding table is not square".to_owned(),
        ));
    }
    let mut output = vec![0.0; grid_height * grid_width * hidden_size];
    let interp = |coordinate: usize, destination: usize| -> (usize, usize, f32) {
        if destination <= 1 {
            return (0, 0, 0.0);
        }
        let value = coordinate as f32 * (source_side - 1) as f32 / (destination - 1) as f32;
        let low = value.floor() as usize;
        (low, (low + 1).min(source_side - 1), value - low as f32)
    };
    for block_y in 0..grid_height / merge_size {
        for block_x in 0..grid_width / merge_size {
            for intra_y in 0..merge_size {
                for intra_x in 0..merge_size {
                    let y = block_y * merge_size + intra_y;
                    let x = block_x * merge_size + intra_x;
                    let (y0, y1, dy) = interp(y, grid_height);
                    let (x0, x1, dx) = interp(x, grid_width);
                    let sources = [
                        (y0 * source_side + x0, (1.0 - dy) * (1.0 - dx)),
                        (y0 * source_side + x1, (1.0 - dy) * dx),
                        (y1 * source_side + x0, dy * (1.0 - dx)),
                        (y1 * source_side + x1, dy * dx),
                    ];
                    let output_index =
                        ((block_y * (grid_width / merge_size) + block_x) * merge_size * merge_size
                            + intra_y * merge_size
                            + intra_x)
                            * hidden_size;
                    for (source, coefficient) in sources {
                        let source_index = source * hidden_size;
                        for feature in 0..hidden_size {
                            output[output_index + feature] +=
                                table[source_index + feature] * coefficient;
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

type VisionQkv = (Vec<f32>, Vec<f32>, Vec<f32>);

fn split_vision_qkv(
    values: &[f32],
    hidden_size: usize,
    num_heads: usize,
    grid_height: usize,
    grid_width: usize,
    merge_size: usize,
) -> Result<VisionQkv, NativeError> {
    if values.len() % (hidden_size * 3) != 0 || hidden_size % num_heads != 0 {
        return Err(NativeError::VectorLengthMismatch {
            actual: values.len(),
            expected: hidden_size * 3,
        });
    }
    let sequence = values.len() / (hidden_size * 3);
    let mut query = Vec::with_capacity(sequence * hidden_size);
    let mut key = Vec::with_capacity(sequence * hidden_size);
    let mut value = Vec::with_capacity(sequence * hidden_size);
    let head_dim = hidden_size / num_heads;
    for (token, row) in values.chunks_exact(hidden_size * 3).enumerate() {
        // Input patches are already permuted into 2x2 spatial-merge blocks by
        // the processor, so derive the original grid coordinate from that
        // block-major order before applying visual RoPE.
        let block_elements = merge_size * merge_size;
        let blocks_per_row = grid_width / merge_size;
        let block_index = token / block_elements;
        let intra = token % block_elements;
        let row_position = (block_index / blocks_per_row) * merge_size + intra / merge_size;
        let column_position = (block_index % blocks_per_row) * merge_size + intra % merge_size;
        for section in 0..3 {
            let target = match section {
                0 => &mut query,
                1 => &mut key,
                _ => &mut value,
            };
            for head in 0..num_heads {
                let offset = section * hidden_size + head * head_dim;
                let mut values = row[offset..offset + head_dim].to_vec();
                if section < 2 {
                    apply_vision_rope(&mut values, row_position, column_position);
                }
                target.extend(values);
            }
        }
    }
    if sequence != grid_height * grid_width {
        return Err(NativeError::Prompt(
            "vision qkv sequence does not match the image grid".to_owned(),
        ));
    }
    Ok((query, key, value))
}

fn apply_vision_rope(values: &mut [f32], row: usize, column: usize) {
    let half = values.len() / 2;
    for index in 0..half {
        let coordinate = if index < half / 2 { row } else { column };
        let local = index % (half / 2).max(1);
        let exponent = (2 * local) as f32 / half.max(1) as f32;
        let angle = coordinate as f32 / 10_000.0_f32.powf(exponent);
        let (sin, cos) = angle.sin_cos();
        let left = values[index];
        let right = values[index + half];
        values[index] = left * cos - right * sin;
        values[index + half] = right * cos + left * sin;
    }
}

fn merge_vision_patches(
    values: &[f32],
    grid_height: usize,
    grid_width: usize,
    hidden_size: usize,
    merge_size: usize,
) -> Result<Vec<f32>, NativeError> {
    if grid_height % merge_size != 0
        || grid_width % merge_size != 0
        || values.len() != grid_height * grid_width * hidden_size
    {
        return Err(NativeError::Prompt(
            "vision features do not align with the configured spatial merge grid".to_owned(),
        ));
    }
    // Preprocessing and positional interpolation already emit block-major
    // patches. The merger's reshape(-1, hidden_size * merge_size^2) therefore
    // only needs the existing contiguous groups, with no second permutation.
    Ok(values.to_vec())
}

struct NativeModel {
    config: TextRuntimeConfig,
    layers: Vec<LayerWeights>,
    embed_tokens: Q4AffineMatrix,
    lm_head: Q4AffineMatrix,
    model_norm: Vec<f32>,
    model_norm_gpu: MetalF32Buffer,
}

#[allow(clippy::large_enum_variant)]
enum LayerWeights {
    Linear(LinearLayerWeights),
    Full(FullLayerWeights),
}

struct CommonLayerWeights {
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    input_norm_gpu: MetalF32Buffer,
    post_attention_norm_gpu: MetalF32Buffer,
    gate_proj: Q4AffineMatrix,
    up_proj: Q4AffineMatrix,
    down_proj: Q4AffineMatrix,
}

struct LinearLayerWeights {
    common: CommonLayerWeights,
    in_proj_qkv: Q4AffineMatrix,
    in_proj_z: Q4AffineMatrix,
    in_proj_b: Q4AffineMatrix,
    in_proj_a: Q4AffineMatrix,
    out_proj: Q4AffineMatrix,
    delta: MetalDeltaNetWeights,
}

struct FullLayerWeights {
    common: CommonLayerWeights,
    q_proj: Q4AffineMatrix,
    k_proj: Q4AffineMatrix,
    v_proj: Q4AffineMatrix,
    o_proj: Q4AffineMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    q_norm_gpu: MetalF32Buffer,
    k_norm_gpu: MetalF32Buffer,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl NativeModel {
    fn load(
        weights: &NativeWeights,
        config: TextRuntimeConfig,
        runtime: &MetalRuntime,
    ) -> Result<Self, NativeError> {
        let embed_tokens = weights.q4_matrix("language_model.model.embed_tokens.weight")?;
        let lm_head = weights.q4_matrix("language_model.lm_head.weight")?;
        let model_norm = weights.tensor_values_f32("language_model.model.norm.weight")?;
        if model_norm.len() != config.hidden_size {
            return Err(NativeError::WrongVectorLength {
                name: "language_model.model.norm.weight".to_owned(),
                actual: model_norm.len(),
                expected: config.hidden_size,
            });
        }
        let model_norm_gpu = runtime
            .create_f32_buffer(&model_norm)
            .map_err(NativeError::Metal)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("language_model.model.layers.{index}");
            let input_norm = load_vector(
                weights,
                &format!("{prefix}.input_layernorm.weight"),
                config.hidden_size,
            )?;
            let post_attention_norm = load_vector(
                weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
                config.hidden_size,
            )?;
            let common = CommonLayerWeights {
                input_norm_gpu: runtime
                    .create_f32_buffer(&input_norm)
                    .map_err(NativeError::Metal)?,
                post_attention_norm_gpu: runtime
                    .create_f32_buffer(&post_attention_norm)
                    .map_err(NativeError::Metal)?,
                input_norm,
                post_attention_norm,
                gate_proj: weights.q4_matrix(&format!("{prefix}.mlp.gate_proj.weight"))?,
                up_proj: weights.q4_matrix(&format!("{prefix}.mlp.up_proj.weight"))?,
                down_proj: weights.q4_matrix(&format!("{prefix}.mlp.down_proj.weight"))?,
            };
            let is_linear = config
                .layer_types
                .get(index)
                .map(|kind| kind == "linear_attention")
                .unwrap_or_else(|| (index + 1) % config.full_attention_interval.max(1) != 0);
            if is_linear {
                let delta_config = DeltaNetConfig {
                    key_heads: config.linear_num_key_heads,
                    value_heads: config.linear_num_value_heads,
                    key_head_dim: config.linear_key_head_dim,
                    value_head_dim: config.linear_value_head_dim,
                    conv_kernel_size: config.linear_conv_kernel_dim,
                };
                let conv_weight = load_vector(
                    weights,
                    &format!("{prefix}.linear_attn.conv1d.weight"),
                    (config.linear_num_key_heads * config.linear_key_head_dim * 2
                        + config.linear_num_value_heads * config.linear_value_head_dim)
                        * config.linear_conv_kernel_dim,
                )?;
                let a_log = load_vector(
                    weights,
                    &format!("{prefix}.linear_attn.A_log"),
                    config.linear_num_value_heads,
                )?;
                let dt_bias = load_vector(
                    weights,
                    &format!("{prefix}.linear_attn.dt_bias"),
                    config.linear_num_value_heads,
                )?;
                let norm = load_vector(
                    weights,
                    &format!("{prefix}.linear_attn.norm.weight"),
                    config.linear_value_head_dim,
                )?;
                let linear = LinearLayerWeights {
                    common,
                    in_proj_qkv: weights
                        .q4_matrix(&format!("{prefix}.linear_attn.in_proj_qkv.weight"))?,
                    in_proj_z: weights
                        .q4_matrix(&format!("{prefix}.linear_attn.in_proj_z.weight"))?,
                    in_proj_b: weights
                        .q4_matrix(&format!("{prefix}.linear_attn.in_proj_b.weight"))?,
                    in_proj_a: weights
                        .q4_matrix(&format!("{prefix}.linear_attn.in_proj_a.weight"))?,
                    out_proj: weights
                        .q4_matrix(&format!("{prefix}.linear_attn.out_proj.weight"))?,
                    delta: runtime
                        .create_deltanet_weights(
                            delta_config,
                            &conv_weight,
                            &a_log,
                            &dt_bias,
                            &norm,
                        )
                        .map_err(NativeError::Metal)?,
                };
                layers.push(LayerWeights::Linear(linear));
            } else {
                let q_norm = load_vector(
                    weights,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    config.head_dim,
                )?;
                let k_norm = load_vector(
                    weights,
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    config.head_dim,
                )?;
                let full = FullLayerWeights {
                    common,
                    q_proj: weights.q4_matrix(&format!("{prefix}.self_attn.q_proj.weight"))?,
                    k_proj: weights.q4_matrix(&format!("{prefix}.self_attn.k_proj.weight"))?,
                    v_proj: weights.q4_matrix(&format!("{prefix}.self_attn.v_proj.weight"))?,
                    o_proj: weights.q4_matrix(&format!("{prefix}.self_attn.o_proj.weight"))?,
                    q_norm_gpu: runtime
                        .create_f32_buffer(&q_norm)
                        .map_err(NativeError::Metal)?,
                    k_norm_gpu: runtime
                        .create_f32_buffer(&k_norm)
                        .map_err(NativeError::Metal)?,
                    q_norm,
                    k_norm,
                    num_attention_heads: config.num_attention_heads,
                    num_key_value_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                };
                layers.push(LayerWeights::Full(full));
            }
        }

        Ok(Self {
            config,
            layers,
            embed_tokens,
            lm_head,
            model_norm,
            model_norm_gpu,
        })
    }

    fn logits(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        hidden: &[f32],
    ) -> Result<Vec<f32>, NativeError> {
        let normalized = rms_norm(hidden, &self.model_norm, self.config.rms_norm_eps)?;
        weights.q4_affine_matvec(runtime, &self.lm_head, &normalized)
    }

    fn argmax_logits_rows(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        hidden: &[f32],
        batch_size: usize,
    ) -> Result<Vec<u32>, NativeError> {
        let expected_hidden = batch_size
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| {
                NativeError::DimensionOverflow("argmax batch hidden elements".to_owned())
            })?;
        if batch_size == 0 || hidden.len() != expected_hidden {
            return Err(NativeError::VectorLengthMismatch {
                actual: hidden.len(),
                expected: expected_hidden,
            });
        }
        let normalized = rms_norm_rows(hidden, &self.model_norm, self.config.rms_norm_eps)?;
        weights.q4_affine_argmax_batch(runtime, &self.lm_head, &normalized, batch_size)
    }

    fn forward_token(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        state: &mut RuntimeState,
        token_id: u32,
        position: MropePosition,
    ) -> Result<Vec<f32>, NativeError> {
        if token_id as usize >= self.config.vocab_size {
            return Err(NativeError::TokenOutOfRange(token_id));
        }
        let hidden = dequantized_row(weights, &self.embed_tokens, token_id as usize)?;
        self.forward_embedding(runtime, weights, state, hidden, position)
    }

    /// Runs a prompt as a layer-major prefill. DeltaNet and causal GQA still
    /// scan positions in order inside each layer, while all independent Q4
    /// projections and MLPs consume the complete prompt matrix at once.
    fn prefill(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        state: &mut RuntimeState,
        token_ids: &[u32],
        positions: &[MropePosition],
        embedding_overrides: &[Option<&[f32]>],
    ) -> Result<Vec<f32>, NativeError> {
        self.prefill_internal(
            runtime,
            weights,
            state,
            token_ids,
            positions,
            embedding_overrides,
            false,
        )
    }

    /// Variant used by speculative verification. It preserves every causal
    /// row's hidden state so the target logits for drafted tokens and the
    /// bonus token can be sampled in one projection batch.
    fn prefill_all(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        state: &mut RuntimeState,
        token_ids: &[u32],
        positions: &[MropePosition],
        embedding_overrides: &[Option<&[f32]>],
    ) -> Result<Vec<f32>, NativeError> {
        self.prefill_internal(
            runtime,
            weights,
            state,
            token_ids,
            positions,
            embedding_overrides,
            true,
        )
    }

    /// Verifies a short speculative block through the GPU-resident batch
    /// graph. The caller must have opened a speculation transaction so linear
    /// layers can write their shadow recurrent states and full-attention KV
    /// caches can be rolled back on rejection.
    fn prefill_verify_gpu(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        state: &mut RuntimeState,
        token_ids: &[u32],
        positions: &[MropePosition],
    ) -> Result<(Vec<f32>, Vec<u32>), NativeError> {
        if token_ids.is_empty() || token_ids.len() != positions.len() {
            return Err(NativeError::Prompt(
                "MTP verify tokens and positions must have matching non-zero lengths".to_owned(),
            ));
        }
        let batch_size = token_ids.len();
        let mut hidden = Vec::with_capacity(
            batch_size
                .checked_mul(self.config.hidden_size)
                .ok_or_else(|| NativeError::DimensionOverflow("MTP verify hidden".to_owned()))?,
        );
        for &token_id in token_ids {
            if token_id as usize >= self.config.vocab_size {
                return Err(NativeError::TokenOutOfRange(token_id));
            }
            hidden.extend_from_slice(&dequantized_row(
                weights,
                &self.embed_tokens,
                token_id as usize,
            )?);
        }
        let mut batch = match state.verify_batch.take() {
            Some(batch) if batch.batch_size == batch_size => batch,
            _ => runtime
                .create_batch_decode_state(self.config.hidden_size, batch_size)
                .map_err(NativeError::Metal)?,
        };
        runtime
            .write_batch_decode_hidden(&mut batch, &hidden)
            .map_err(NativeError::Metal)?;
        let positions = positions
            .iter()
            .map(|position| position.0)
            .collect::<Vec<_>>();
        let workspace = state.speculation.as_ref().ok_or_else(|| {
            NativeError::InvalidConfig("MTP verify graph requires active speculation".to_owned())
        })?;
        let rope = self.config.rope();
        let lm_head_job = weights
            .mapped_q4_jobs(&[&self.lm_head], self.config.hidden_size)?
            .into_iter()
            .next()
            .ok_or_else(|| NativeError::InvalidConfig("MTP LM head mapping is empty".to_owned()))?;
        let mut graph = Vec::with_capacity(self.layers.len());
        let mut layer_states = state.layers.iter_mut();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_state = layer_states.next().ok_or_else(|| {
                NativeError::InvalidConfig("MTP verify layer state is incomplete".to_owned())
            })?;
            match (layer, layer_state) {
                (LayerWeights::Linear(linear), LayerRuntimeState::Linear(active)) => {
                    let shadow =
                        workspace.shadow_linear[layer_index]
                            .as_ref()
                            .ok_or_else(|| {
                                NativeError::InvalidConfig(
                                    "MTP verify DeltaNet shadow is missing".to_owned(),
                                )
                            })?;
                    let descriptor = linear.gpu_decode_layer(weights, shadow)?;
                    graph.push(MetalBatchDecodeLayer::Linear {
                        layer: descriptor,
                        source: &*active,
                        destination: shadow,
                        snapshots: workspace.snapshots[layer_index].as_ref(),
                    });
                }
                (LayerWeights::Full(full), LayerRuntimeState::Full(kv_state)) => {
                    let descriptor =
                        full.gpu_decode_layer(weights, MropePosition(positions[0]), &rope)?;
                    graph.push(MetalBatchDecodeLayer::Full(descriptor, kv_state));
                }
                _ => unreachable!("layer weights and runtime state are constructed together"),
            }
        }
        let result = runtime
            .decode_batch_layers_with_argmax(
                &mut batch,
                &mut graph,
                &positions,
                self.config.rms_norm_eps,
                &self.model_norm_gpu,
                &lm_head_job,
            )
            .map_err(NativeError::Metal);
        state.verify_batch = Some(batch);
        result
    }

    /// Fuses the default one-draft MTP round with its next adapter seed. The
    /// target verifier still owns acceptance and output tokens; the adapter
    /// only consumes the selected target row after its greedy result exists.
    #[allow(clippy::too_many_arguments)]
    fn prefill_verify_mtp_seed_gpu(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        state: &mut RuntimeState,
        token_ids: &[u32],
        positions: &[MropePosition],
        adapter: &MtpAdapter,
        adapter_state: &mut MtpRequestState,
        draft_token: u32,
    ) -> Result<MetalMtpVerifyResult, NativeError> {
        if token_ids.len() != 2 || positions.len() != token_ids.len() {
            return Err(NativeError::Prompt(
                "fused MTP verification requires exactly one draft token".to_owned(),
            ));
        }
        if token_ids[1] != draft_token {
            return Err(NativeError::InvalidConfig(
                "fused MTP draft token does not match the verifier row".to_owned(),
            ));
        }
        let mut hidden = Vec::with_capacity(
            token_ids
                .len()
                .checked_mul(self.config.hidden_size)
                .ok_or_else(|| {
                    NativeError::DimensionOverflow("fused MTP verify hidden".to_owned())
                })?,
        );
        for &token_id in token_ids {
            if token_id as usize >= self.config.vocab_size {
                return Err(NativeError::TokenOutOfRange(token_id));
            }
            hidden.extend_from_slice(&dequantized_row(
                weights,
                &self.embed_tokens,
                token_id as usize,
            )?);
        }
        let batch_size = token_ids.len();
        let mut batch = match state.verify_batch.take() {
            Some(batch) if batch.batch_size == batch_size => batch,
            _ => runtime
                .create_batch_decode_state(self.config.hidden_size, batch_size)
                .map_err(NativeError::Metal)?,
        };
        runtime
            .write_batch_decode_hidden(&mut batch, &hidden)
            .map_err(NativeError::Metal)?;
        let positions = positions
            .iter()
            .map(|position| position.0)
            .collect::<Vec<_>>();
        let workspace = state.speculation.as_ref().ok_or_else(|| {
            NativeError::InvalidConfig("MTP verify graph requires active speculation".to_owned())
        })?;
        let rope = self.config.rope();
        let lm_head_job = weights
            .mapped_q4_jobs(&[&self.lm_head], self.config.hidden_size)?
            .into_iter()
            .next()
            .ok_or_else(|| NativeError::InvalidConfig("MTP LM head mapping is empty".to_owned()))?;
        let embedding_job = weights
            .mapped_q4_jobs(&[&self.embed_tokens], self.config.hidden_size)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                NativeError::InvalidConfig("MTP embedding mapping is empty".to_owned())
            })?;
        let adapter_fc = adapter
            .weights
            .mapped_q4_jobs(&[&adapter.fc], self.config.hidden_size * 2)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                NativeError::InvalidConfig("MTP adapter FC mapping is empty".to_owned())
            })?;
        let adapter_rope = adapter.config.rope();
        let adapter_layer = adapter.layer.gpu_decode_layer(
            &adapter.weights,
            MropePosition::text(adapter_state.next_position),
            &adapter_rope,
        )?;
        let mut graph = Vec::with_capacity(self.layers.len());
        let mut layer_states = state.layers.iter_mut();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_state = layer_states.next().ok_or_else(|| {
                NativeError::InvalidConfig("MTP verify layer state is incomplete".to_owned())
            })?;
            match (layer, layer_state) {
                (LayerWeights::Linear(linear), LayerRuntimeState::Linear(active)) => {
                    let shadow =
                        workspace.shadow_linear[layer_index]
                            .as_ref()
                            .ok_or_else(|| {
                                NativeError::InvalidConfig(
                                    "MTP verify DeltaNet shadow is missing".to_owned(),
                                )
                            })?;
                    let descriptor = linear.gpu_decode_layer(weights, shadow)?;
                    graph.push(MetalBatchDecodeLayer::Linear {
                        layer: descriptor,
                        source: &*active,
                        destination: shadow,
                        snapshots: workspace.snapshots[layer_index].as_ref(),
                    });
                }
                (LayerWeights::Full(full), LayerRuntimeState::Full(kv_state)) => {
                    let descriptor =
                        full.gpu_decode_layer(weights, MropePosition(positions[0]), &rope)?;
                    graph.push(MetalBatchDecodeLayer::Full(descriptor, kv_state));
                }
                _ => unreachable!("layer weights and runtime state are constructed together"),
            }
        }
        let result = runtime
            .decode_batch_layers_with_mtp_seed(
                &mut batch,
                &mut graph,
                &positions,
                self.config.rms_norm_eps,
                &self.model_norm_gpu,
                &lm_head_job,
                &mut adapter_state.decode,
                &embedding_job,
                &adapter.pre_fc_norm_embedding_gpu,
                &adapter.pre_fc_norm_hidden_gpu,
                &adapter_fc,
                &adapter_layer,
                adapter.mtp_mlp_f16.as_ref(),
                &mut adapter_state.kv,
                &adapter.norm_gpu,
                adapter.config.rms_norm_eps,
                draft_token,
            )
            .map_err(NativeError::Metal);
        state.verify_batch = Some(batch);
        match result {
            Ok(result) => {
                adapter_state.next_position = adapter_state.next_position.saturating_add(1);
                adapter_state.seed_token = Some(result.seed_token);
                adapter_state.seed_hidden = None;
                adapter_state.round_appended = 0;
                Ok(result)
            }
            Err(error) => Err(error),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_internal(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        state: &mut RuntimeState,
        token_ids: &[u32],
        positions: &[MropePosition],
        embedding_overrides: &[Option<&[f32]>],
        return_all: bool,
    ) -> Result<Vec<f32>, NativeError> {
        if token_ids.is_empty()
            || token_ids.len() != positions.len()
            || token_ids.len() != embedding_overrides.len()
        {
            return Err(NativeError::Prompt(
                "prefill tokens, positions, and embedding overrides must have matching lengths"
                    .to_owned(),
            ));
        }
        state.reserve_prefill(runtime, token_ids.len())?;
        if token_ids.len() == 1 {
            return match embedding_overrides[0] {
                Some(embedding) => self.forward_embedding(
                    runtime,
                    weights,
                    state,
                    embedding.to_vec(),
                    positions[0],
                ),
                None => self.forward_token(runtime, weights, state, token_ids[0], positions[0]),
            };
        }

        let batch_size = token_ids.len();
        let mut hidden =
            Vec::with_capacity(batch_size.checked_mul(self.config.hidden_size).ok_or_else(
                || NativeError::DimensionOverflow("prefill hidden activations".to_owned()),
            )?);
        for (token_id, embedding) in token_ids
            .iter()
            .copied()
            .zip(embedding_overrides.iter().copied())
        {
            let row = match embedding {
                Some(embedding) => embedding,
                None => {
                    if token_id as usize >= self.config.vocab_size {
                        return Err(NativeError::TokenOutOfRange(token_id));
                    }
                    let embedding =
                        dequantized_row(weights, &self.embed_tokens, token_id as usize)?;
                    hidden.extend_from_slice(&embedding);
                    continue;
                }
            };
            if row.len() != self.config.hidden_size {
                return Err(NativeError::VectorLengthMismatch {
                    actual: row.len(),
                    expected: self.config.hidden_size,
                });
            }
            hidden.extend_from_slice(row);
        }

        let profile = std::env::var_os("QWEN38_PROFILE").is_some();
        let prefill_started = profile.then(Instant::now);
        let prefill_chunk_size = configured_prefill_chunk_tokens(batch_size)?;
        let rope = self.config.rope();
        let (layer_states, speculation) = (&mut state.layers, &mut state.speculation);
        for (layer_index, (layer, layer_state)) in
            self.layers.iter().zip(layer_states.iter_mut()).enumerate()
        {
            let layer_started = profile.then(Instant::now);
            let mut norm_elapsed = Duration::ZERO;
            let mut mixed_elapsed = Duration::ZERO;
            let mut residual_elapsed = Duration::ZERO;
            let mut post_norm_elapsed = Duration::ZERO;
            let mut mlp_elapsed = Duration::ZERO;
            let mut mlp_residual_elapsed = Duration::ZERO;
            let (gate_proj, up_proj, down_proj) = layer.mlp_projections();
            for chunk_start in (0..batch_size).step_by(prefill_chunk_size) {
                let first_chunk = chunk_start == 0;
                let chunk_end = (chunk_start + prefill_chunk_size).min(batch_size);
                let chunk_tokens = chunk_end - chunk_start;
                let hidden_start = chunk_start
                    .checked_mul(self.config.hidden_size)
                    .ok_or_else(|| {
                        NativeError::DimensionOverflow("prefill chunk start".to_owned())
                    })?;
                let hidden_end =
                    chunk_end
                        .checked_mul(self.config.hidden_size)
                        .ok_or_else(|| {
                            NativeError::DimensionOverflow("prefill chunk end".to_owned())
                        })?;

                let norm_started = profile.then(Instant::now);
                let normalized = rms_norm_rows(
                    &hidden[hidden_start..hidden_end],
                    layer.input_norm(),
                    self.config.rms_norm_eps,
                )?;
                if let Some(started) = norm_started {
                    norm_elapsed += started.elapsed();
                }
                let mixed_started = profile.then(Instant::now);
                let mixed = match (layer, &mut *layer_state) {
                    (LayerWeights::Linear(linear), LayerRuntimeState::Linear(layer_state)) => {
                        if let Some(workspace) =
                            speculation.as_mut().filter(|workspace| workspace.active)
                        {
                            let shadow =
                                workspace.shadow_linear[layer_index]
                                    .as_mut()
                                    .ok_or_else(|| {
                                        NativeError::InvalidConfig(
                                            "missing DeltaNet speculation shadow".to_owned(),
                                        )
                                    })?;
                            if first_chunk {
                                linear.forward_prefill_from(
                                    runtime,
                                    weights,
                                    &normalized,
                                    chunk_tokens,
                                    layer_state,
                                    shadow,
                                    self.config.rms_norm_eps,
                                )?
                            } else {
                                linear.forward_prefill(
                                    runtime,
                                    weights,
                                    &normalized,
                                    chunk_tokens,
                                    shadow,
                                    self.config.rms_norm_eps,
                                )?
                            }
                        } else {
                            linear.forward_prefill(
                                runtime,
                                weights,
                                &normalized,
                                chunk_tokens,
                                layer_state,
                                self.config.rms_norm_eps,
                            )?
                        }
                    }
                    (LayerWeights::Full(full), LayerRuntimeState::Full(layer_state)) => full
                        .forward_prefill(
                            runtime,
                            weights,
                            &normalized,
                            &positions[chunk_start..chunk_end],
                            layer_state,
                            &rope,
                            self.config.rms_norm_eps,
                        )?,
                    _ => unreachable!("layer weights and runtime state are constructed together"),
                };
                if let Some(started) = mixed_started {
                    mixed_elapsed += started.elapsed();
                }
                let residual_started = profile.then(Instant::now);
                add_in_place(&mut hidden[hidden_start..hidden_end], &mixed)?;
                if let Some(started) = residual_started {
                    residual_elapsed += started.elapsed();
                }
                let post_norm_started = profile.then(Instant::now);
                let post_norm = rms_norm_rows(
                    &hidden[hidden_start..hidden_end],
                    layer.post_attention_norm(),
                    self.config.rms_norm_eps,
                )?;
                if let Some(started) = post_norm_started {
                    post_norm_elapsed += started.elapsed();
                }
                let mlp_started = profile.then(Instant::now);
                let mlp = weights.q4_affine_mlp_batch(
                    runtime,
                    gate_proj,
                    up_proj,
                    down_proj,
                    &post_norm,
                    chunk_tokens,
                )?;
                if let Some(started) = mlp_started {
                    mlp_elapsed += started.elapsed();
                }
                let mlp_residual_started = profile.then(Instant::now);
                add_in_place(&mut hidden[hidden_start..hidden_end], &mlp)?;
                if let Some(started) = mlp_residual_started {
                    mlp_residual_elapsed += started.elapsed();
                }
            }
            if let Some(layer_started) = layer_started {
                eprintln!(
                    "prefill layer={layer_index} total={:.3}ms norm={:.3}ms mixed={:.3}ms residual={:.3}ms post_norm={:.3}ms mlp={:.3}ms mlp_residual={:.3}ms",
                    layer_started.elapsed().as_secs_f64() * 1_000.0,
                    norm_elapsed.as_secs_f64() * 1_000.0,
                    mixed_elapsed.as_secs_f64() * 1_000.0,
                    residual_elapsed.as_secs_f64() * 1_000.0,
                    post_norm_elapsed.as_secs_f64() * 1_000.0,
                    mlp_elapsed.as_secs_f64() * 1_000.0,
                    mlp_residual_elapsed.as_secs_f64() * 1_000.0,
                );
            }
        }
        if let Some(prefill_started) = prefill_started {
            eprintln!(
                "prefill total={:.3}ms tokens={batch_size} tok_per_s={:.3}",
                prefill_started.elapsed().as_secs_f64() * 1_000.0,
                batch_size as f64 / prefill_started.elapsed().as_secs_f64(),
            );
        }
        if return_all {
            return Ok(hidden);
        }
        let final_offset = hidden
            .len()
            .checked_sub(self.config.hidden_size)
            .ok_or_else(|| {
                NativeError::DimensionOverflow("prefill final hidden activation".to_owned())
            })?;
        Ok(hidden[final_offset..].to_vec())
    }

    fn forward_embedding(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        state: &mut RuntimeState,
        mut hidden: Vec<f32>,
        position: MropePosition,
    ) -> Result<Vec<f32>, NativeError> {
        if hidden.len() != self.config.hidden_size {
            return Err(NativeError::VectorLengthMismatch {
                actual: hidden.len(),
                expected: self.config.hidden_size,
            });
        }
        let profile = std::env::var_os("QWEN38_PROFILE").is_some();
        let gpu_decode = std::env::var_os("QWEN38_DISABLE_GPU_DECODE").is_none();
        let gpu_decode_graph =
            gpu_decode && std::env::var_os("QWEN38_DISABLE_GPU_DECODE_GRAPH").is_none();
        let decode_started = profile.then(Instant::now);
        let mut norm_elapsed = Duration::ZERO;
        let mut mixed_elapsed = Duration::ZERO;
        let mut residual_elapsed = Duration::ZERO;
        let mut post_norm_elapsed = Duration::ZERO;
        let mut mlp_elapsed = Duration::ZERO;
        let mut gpu_linear_elapsed = Duration::ZERO;
        let mut gpu_full_elapsed = Duration::ZERO;
        let rope = self.config.rope();
        let (layer_states, decode) = (&mut state.layers, &mut state.decode);
        if gpu_decode_graph {
            runtime
                .write_decode_hidden(decode, &hidden)
                .map_err(NativeError::Metal)?;
            let started = profile.then(Instant::now);
            let mut gpu_layers = Vec::with_capacity(self.layers.len());
            for (layer, layer_state) in self.layers.iter().zip(layer_states.iter_mut()) {
                match (layer, layer_state) {
                    (LayerWeights::Linear(linear), LayerRuntimeState::Linear(linear_state)) => {
                        gpu_layers.push(MetalDecodeLayer::Linear(
                            linear.gpu_decode_layer(weights, linear_state)?,
                        ));
                    }
                    (LayerWeights::Full(full), LayerRuntimeState::Full(kv_state)) => {
                        gpu_layers.push(MetalDecodeLayer::Full(
                            full.gpu_decode_layer(weights, position, &rope)?,
                            kv_state,
                        ));
                    }
                    _ => unreachable!("layer weights and runtime state are constructed together"),
                }
            }
            runtime
                .decode_layers(decode, &mut gpu_layers, self.config.rms_norm_eps)
                .map_err(NativeError::Metal)?;
            hidden = runtime
                .read_decode_hidden(decode)
                .map_err(NativeError::Metal)?;
            if let Some(started) = started {
                eprintln!(
                    "decode total={:.3}ms gpu_graph={:.3}ms layers={}",
                    decode_started
                        .expect("decode profiling start exists with graph timing")
                        .elapsed()
                        .as_secs_f64()
                        * 1_000.0,
                    started.elapsed().as_secs_f64() * 1_000.0,
                    self.layers.len(),
                );
            }
            return Ok(hidden);
        }
        if gpu_decode {
            runtime
                .write_decode_hidden(decode, &hidden)
                .map_err(NativeError::Metal)?;
        }
        let mut layer_index = 0;
        while layer_index < self.layers.len() {
            if gpu_decode {
                if matches!(&self.layers[layer_index], LayerWeights::Linear(_)) {
                    let started = profile.then(Instant::now);
                    let mut gpu_layers = Vec::new();
                    while layer_index < self.layers.len() {
                        let LayerWeights::Linear(linear) = &self.layers[layer_index] else {
                            break;
                        };
                        let LayerRuntimeState::Linear(layer_state) = &layer_states[layer_index]
                        else {
                            unreachable!(
                                "layer weights and runtime state are constructed together"
                            );
                        };
                        gpu_layers.push(linear.gpu_decode_layer(weights, layer_state)?);
                        layer_index += 1;
                    }
                    runtime
                        .decode_linear_layers(decode, &gpu_layers, self.config.rms_norm_eps)
                        .map_err(NativeError::Metal)?;
                    if let Some(started) = started {
                        gpu_linear_elapsed += started.elapsed();
                    }
                    continue;
                }
                if let LayerWeights::Full(full) = &self.layers[layer_index] {
                    let LayerRuntimeState::Full(layer_state) = &mut layer_states[layer_index]
                    else {
                        unreachable!("layer weights and runtime state are constructed together");
                    };
                    let started = profile.then(Instant::now);
                    let gpu_layer = full.gpu_decode_layer(weights, position, &rope)?;
                    runtime
                        .decode_full_layer(
                            decode,
                            &gpu_layer,
                            layer_state,
                            self.config.rms_norm_eps,
                        )
                        .map_err(NativeError::Metal)?;
                    if let Some(started) = started {
                        gpu_full_elapsed += started.elapsed();
                    }
                    layer_index += 1;
                    continue;
                }
                hidden = runtime
                    .read_decode_hidden(decode)
                    .map_err(NativeError::Metal)?;
            }
            let layer = &self.layers[layer_index];
            let layer_state = &mut layer_states[layer_index];
            let started = profile.then(Instant::now);
            let normalized = rms_norm(&hidden, layer.input_norm(), self.config.rms_norm_eps)?;
            if let Some(started) = started {
                norm_elapsed += started.elapsed();
            }
            let started = profile.then(Instant::now);
            let mixed = match (layer, layer_state) {
                (LayerWeights::Linear(linear), LayerRuntimeState::Linear(layer_state)) => linear
                    .forward(
                        runtime,
                        weights,
                        &normalized,
                        layer_state,
                        self.config.rms_norm_eps,
                    )?,
                (LayerWeights::Full(full), LayerRuntimeState::Full(layer_state)) => full.forward(
                    runtime,
                    weights,
                    &normalized,
                    layer_state,
                    position,
                    rope.clone(),
                    self.config.rms_norm_eps,
                )?,
                _ => unreachable!("layer weights and runtime state are constructed together"),
            };
            if let Some(started) = started {
                mixed_elapsed += started.elapsed();
            }
            let started = profile.then(Instant::now);
            add_in_place(&mut hidden, &mixed)?;
            if let Some(started) = started {
                residual_elapsed += started.elapsed();
            }
            let started = profile.then(Instant::now);
            let post_norm = rms_norm(
                &hidden,
                layer.post_attention_norm(),
                self.config.rms_norm_eps,
            )?;
            if let Some(started) = started {
                post_norm_elapsed += started.elapsed();
            }
            let (gate_proj, up_proj, down_proj) = layer.mlp_projections();
            let started = profile.then(Instant::now);
            let mlp = weights
                .q4_affine_mlp_batch(runtime, gate_proj, up_proj, down_proj, &post_norm, 1)?;
            if let Some(started) = started {
                mlp_elapsed += started.elapsed();
            }
            let started = profile.then(Instant::now);
            add_in_place(&mut hidden, &mlp)?;
            if let Some(started) = started {
                residual_elapsed += started.elapsed();
            }
            if gpu_decode {
                runtime
                    .write_decode_hidden(decode, &hidden)
                    .map_err(NativeError::Metal)?;
            }
            layer_index += 1;
        }
        if gpu_decode {
            hidden = runtime
                .read_decode_hidden(decode)
                .map_err(NativeError::Metal)?;
        }
        if let Some(decode_started) = decode_started {
            eprintln!(
                "decode total={:.3}ms norm={:.3}ms mixed={:.3}ms residual={:.3}ms post_norm={:.3}ms mlp={:.3}ms linear_gpu={:.3}ms full_gpu={:.3}ms",
                decode_started.elapsed().as_secs_f64() * 1_000.0,
                norm_elapsed.as_secs_f64() * 1_000.0,
                mixed_elapsed.as_secs_f64() * 1_000.0,
                residual_elapsed.as_secs_f64() * 1_000.0,
                post_norm_elapsed.as_secs_f64() * 1_000.0,
                mlp_elapsed.as_secs_f64() * 1_000.0,
                gpu_linear_elapsed.as_secs_f64() * 1_000.0,
                gpu_full_elapsed.as_secs_f64() * 1_000.0,
            );
        }
        Ok(hidden)
    }
}

#[derive(Debug, Deserialize)]
struct MtpAdapterFileConfig {
    model_type: String,
    block_size: usize,
    text_config: TextRuntimeConfig,
}

/// The standalone Qwen3.5 MTP export contains a small one-layer drafter. It
/// reuses the target model's embedding table and LM head; only `fc`, the
/// adapter transformer layer, and its norms are owned here.
struct MtpAdapter {
    weights: NativeWeights,
    support: MtpSupport,
    block_size: usize,
    config: TextRuntimeConfig,
    fc: Q4AffineMatrix,
    pre_fc_norm_embedding: Vec<f32>,
    pre_fc_norm_hidden: Vec<f32>,
    pre_fc_norm_embedding_gpu: MetalF32Buffer,
    pre_fc_norm_hidden_gpu: MetalF32Buffer,
    layer: FullLayerWeights,
    norm: Vec<f32>,
    norm_gpu: MetalF32Buffer,
    mtp_mlp_f16: Option<MetalMtpMlpF16>,
}

struct MtpRequestState {
    kv: Q8KvState,
    decode: MetalDecodeState,
    next_position: u32,
    seed_token: Option<u32>,
    seed_hidden: Option<Vec<f32>>,
    round_appended: usize,
}

struct MtpForwardResult {
    hidden: Vec<f32>,
    logits: Option<Vec<f32>>,
    token: Option<u32>,
}

impl MtpAdapter {
    fn load(
        path: &Path,
        target_config: &TextRuntimeConfig,
        runtime: &MetalRuntime,
    ) -> Result<Self, NativeError> {
        let config_path = path.join("config.json");
        let config_bytes =
            std::fs::read(&config_path).map_err(|source| NativeError::ConfigRead {
                path: config_path,
                source,
            })?;
        let file_config: MtpAdapterFileConfig =
            serde_json::from_slice(&config_bytes).map_err(NativeError::ConfigJson)?;
        if !file_config.model_type.eq_ignore_ascii_case("qwen3_5_mtp") {
            return Err(NativeError::InvalidConfig(format!(
                "MTP adapter model_type must be qwen3_5_mtp, got {:?}",
                file_config.model_type
            )));
        }
        if file_config.block_size < 2 || file_config.block_size > u8::MAX as usize {
            return Err(NativeError::InvalidConfig(format!(
                "MTP adapter block_size must be between 2 and {}, got {}",
                u8::MAX,
                file_config.block_size
            )));
        }
        if file_config.text_config.mtp_num_hidden_layers != 1 {
            return Err(NativeError::InvalidConfig(format!(
                "native MTP currently supports one adapter layer, got {}",
                file_config.text_config.mtp_num_hidden_layers
            )));
        }
        validate_runtime_config(&file_config.text_config)?;
        validate_mtp_pair(target_config, &file_config.text_config)?;

        let weights = NativeWeights::open(path, runtime)?;
        let inspection = inspect_model_dir(path).map_err(NativeError::Preflight)?;
        if !matches!(inspection.mtp_support, MtpSupport::Available { .. }) {
            return Err(NativeError::InvalidConfig(format!(
                "MTP adapter weights are incomplete: {}",
                inspection.mtp_support
            )));
        }

        let config = file_config.text_config;
        let hidden_size = config.hidden_size;
        let fc = weights.q4_matrix("fc.weight")?;
        let expected_fc_input = hidden_size
            .checked_mul(2)
            .ok_or_else(|| NativeError::DimensionOverflow("MTP fc input".to_owned()))?;
        if fc.input_elements != expected_fc_input as u64 || fc.output_rows != hidden_size as u64 {
            return Err(NativeError::InvalidConfig(format!(
                "MTP fc shape is {}x{}, expected {}x{}",
                fc.output_rows, fc.input_elements, hidden_size, expected_fc_input
            )));
        }
        let pre_fc_norm_embedding =
            load_vector(&weights, "pre_fc_norm_embedding.weight", hidden_size)?;
        let pre_fc_norm_hidden = load_vector(&weights, "pre_fc_norm_hidden.weight", hidden_size)?;
        let pre_fc_norm_embedding_gpu = runtime
            .create_f32_buffer(&pre_fc_norm_embedding)
            .map_err(NativeError::Metal)?;
        let pre_fc_norm_hidden_gpu = runtime
            .create_f32_buffer(&pre_fc_norm_hidden)
            .map_err(NativeError::Metal)?;
        let norm = load_vector(&weights, "norm.weight", hidden_size)?;
        let norm_gpu = runtime
            .create_f32_buffer(&norm)
            .map_err(NativeError::Metal)?;

        let prefix = "layers.0";
        let input_norm = load_vector(
            &weights,
            &format!("{prefix}.input_layernorm.weight"),
            hidden_size,
        )?;
        let post_attention_norm = load_vector(
            &weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden_size,
        )?;
        let common = CommonLayerWeights {
            input_norm_gpu: runtime
                .create_f32_buffer(&input_norm)
                .map_err(NativeError::Metal)?,
            post_attention_norm_gpu: runtime
                .create_f32_buffer(&post_attention_norm)
                .map_err(NativeError::Metal)?,
            input_norm,
            post_attention_norm,
            gate_proj: weights.q4_matrix(&format!("{prefix}.mlp.gate_proj.weight"))?,
            up_proj: weights.q4_matrix(&format!("{prefix}.mlp.up_proj.weight"))?,
            down_proj: weights.q4_matrix(&format!("{prefix}.mlp.down_proj.weight"))?,
        };
        let q_norm = load_vector(
            &weights,
            &format!("{prefix}.self_attn.q_norm.weight"),
            config.head_dim,
        )?;
        let k_norm = load_vector(
            &weights,
            &format!("{prefix}.self_attn.k_norm.weight"),
            config.head_dim,
        )?;
        let layer = FullLayerWeights {
            common,
            q_proj: weights.q4_matrix(&format!("{prefix}.self_attn.q_proj.weight"))?,
            k_proj: weights.q4_matrix(&format!("{prefix}.self_attn.k_proj.weight"))?,
            v_proj: weights.q4_matrix(&format!("{prefix}.self_attn.v_proj.weight"))?,
            o_proj: weights.q4_matrix(&format!("{prefix}.self_attn.o_proj.weight"))?,
            q_norm_gpu: runtime
                .create_f32_buffer(&q_norm)
                .map_err(NativeError::Metal)?,
            k_norm_gpu: runtime
                .create_f32_buffer(&k_norm)
                .map_err(NativeError::Metal)?,
            q_norm,
            k_norm,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
        };
        let mtp_mlp_f16 = if std::env::var_os("QWEN38_ENABLE_MTP_FP16_MLP").is_some()
            && std::env::var_os("QWEN38_DISABLE_MTP_FP16_MLP").is_none()
            && runtime.mps_available()
        {
            let gate_up_jobs = weights.mapped_q4_jobs(
                &[&layer.common.gate_proj, &layer.common.up_proj],
                hidden_size,
            )?;
            let down_jobs = weights.mapped_q4_jobs(
                &[&layer.common.down_proj],
                layer.common.gate_proj.output_rows as usize,
            )?;
            match runtime.create_mtp_mlp_f16(
                hidden_size,
                &gate_up_jobs[0],
                &gate_up_jobs[1],
                &down_jobs[0],
            ) {
                Ok(mlp) => {
                    eprintln!("MTP adapter MLP: persistent FP16 path enabled");
                    Some(mlp)
                }
                Err(error) => {
                    eprintln!("MTP adapter MLP: FP16 path unavailable, using Q4 ({error})");
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            weights,
            support: inspection.mtp_support,
            block_size: file_config.block_size,
            config,
            fc,
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            pre_fc_norm_embedding_gpu,
            pre_fc_norm_hidden_gpu,
            layer,
            norm,
            norm_gpu,
            mtp_mlp_f16,
        })
    }

    fn new_request_state(
        &self,
        runtime: &MetalRuntime,
        next_position: u32,
    ) -> Result<MtpRequestState, NativeError> {
        Ok(MtpRequestState {
            kv: runtime
                .create_q8_kv_state(self.layer.num_key_value_heads, self.layer.head_dim)
                .map_err(NativeError::Metal)?,
            decode: runtime
                .create_decode_state(self.config.hidden_size)
                .map_err(NativeError::Metal)?,
            next_position,
            seed_token: None,
            seed_hidden: None,
            round_appended: 0,
        })
    }

    /// Replays the target prompt into the adapter's attention cache. MTP uses
    /// the shifted pairs `(x[1], h[0]) .. (x[N-1], h[N-2])` and pairs the
    /// target's first bonus token with the final prompt hidden `h[N-1]`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_prompt(
        &self,
        runtime: &MetalRuntime,
        target_weights: &NativeWeights,
        target_model: &NativeModel,
        state: &mut MtpRequestState,
        prompt_ids: &[u32],
        _prompt_positions: &[MropePosition],
        prompt_hidden_rows: &[f32],
        bonus_token: u32,
    ) -> Result<(), NativeError> {
        if prompt_ids.is_empty() || prompt_ids.len() != _prompt_positions.len() {
            return Err(NativeError::Prompt(
                "MTP prompt tokens and positions must have matching non-zero lengths".to_owned(),
            ));
        }
        let hidden_size = self.config.hidden_size;
        let expected_hidden = prompt_ids
            .len()
            .checked_mul(hidden_size)
            .ok_or_else(|| NativeError::DimensionOverflow("MTP prompt hidden rows".to_owned()))?;
        if prompt_hidden_rows.len() != expected_hidden {
            return Err(NativeError::VectorLengthMismatch {
                actual: prompt_hidden_rows.len(),
                expected: expected_hidden,
            });
        }

        // MTP consumes shifted prompt tokens and the target's first bonus:
        //   (x[1], h[0]), ..., (x[N-1], h[N-2]), (bonus, h[N-1]).
        // Its cache has an independent position space starting at zero.
        let row_count = prompt_ids.len();
        let chunk_size = configured_prefill_chunk_tokens(row_count)?;
        state.next_position = 0;
        for chunk_start in (0..row_count).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(row_count);
            let chunk_tokens = chunk_end - chunk_start;
            let mut fc_input =
                Vec::with_capacity(chunk_tokens.checked_mul(hidden_size * 2).ok_or_else(|| {
                    NativeError::DimensionOverflow("MTP prompt FC activations".to_owned())
                })?);
            for row_index in chunk_start..chunk_end {
                let token_id = prompt_ids
                    .get(row_index + 1)
                    .copied()
                    .unwrap_or(bonus_token);
                let embedding = dequantized_row(
                    target_weights,
                    &target_model.embed_tokens,
                    token_id as usize,
                )?;
                let embedding = rms_norm(
                    &embedding,
                    &self.pre_fc_norm_embedding,
                    self.config.rms_norm_eps,
                )?;
                let hidden_start = row_index.checked_mul(hidden_size).ok_or_else(|| {
                    NativeError::DimensionOverflow("MTP prompt hidden offset".to_owned())
                })?;
                let hidden_end = hidden_start.checked_add(hidden_size).ok_or_else(|| {
                    NativeError::DimensionOverflow("MTP prompt hidden end".to_owned())
                })?;
                let hidden = prompt_hidden_rows.get(hidden_start..hidden_end).ok_or(
                    NativeError::VectorLengthMismatch {
                        actual: prompt_hidden_rows.len(),
                        expected: expected_hidden,
                    },
                )?;
                let hidden = rms_norm(hidden, &self.pre_fc_norm_hidden, self.config.rms_norm_eps)?;
                fc_input.extend_from_slice(&embedding);
                fc_input.extend_from_slice(&hidden);
            }

            let mut projected = self
                .weights
                .q4_affine_matmul_batch(runtime, &[&self.fc], &fc_input, chunk_tokens)?
                .remove(0);
            ensure_batched_width(
                &projected,
                chunk_tokens,
                hidden_size,
                "MTP prompt FC projection",
            )?;
            let positions = (chunk_start..chunk_end)
                .map(|position| MropePosition::text(position as u32))
                .collect::<Vec<_>>();
            let attention = self.layer.forward_prefill(
                runtime,
                &self.weights,
                &projected,
                &positions,
                &mut state.kv,
                &self.config.rope(),
                self.config.rms_norm_eps,
            )?;
            add_in_place(&mut projected, &attention)?;
            let post_norm = rms_norm_rows(
                &projected,
                &self.layer.common.post_attention_norm,
                self.config.rms_norm_eps,
            )?;
            let mlp = self.weights.q4_affine_mlp_batch(
                runtime,
                &self.layer.common.gate_proj,
                &self.layer.common.up_proj,
                &self.layer.common.down_proj,
                &post_norm,
                chunk_tokens,
            )?;
            add_in_place(&mut projected, &mlp)?;

            if chunk_end == row_count {
                let final_offset =
                    (chunk_tokens - 1).checked_mul(hidden_size).ok_or_else(|| {
                        NativeError::DimensionOverflow("MTP seed hidden offset".to_owned())
                    })?;
                let final_end = final_offset.checked_add(hidden_size).ok_or_else(|| {
                    NativeError::DimensionOverflow("MTP seed hidden end".to_owned())
                })?;
                let final_hidden = projected.get(final_offset..final_end).ok_or_else(|| {
                    NativeError::VectorLengthMismatch {
                        actual: projected.len(),
                        expected: chunk_tokens * hidden_size,
                    }
                })?;
                let seed_hidden = rms_norm(final_hidden, &self.norm, self.config.rms_norm_eps)?;
                let seed_token = target_weights
                    .q4_affine_argmax_batch(runtime, &target_model.lm_head, &seed_hidden, 1)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        NativeError::InvalidConfig("target argmax returned no token".to_owned())
                    })?;
                state.seed_token = Some(seed_token);
                state.seed_hidden = Some(seed_hidden);
            }
        }
        state.next_position = u32::try_from(row_count)
            .map_err(|_| NativeError::DimensionOverflow("MTP prompt position".to_owned()))?;
        state.round_appended = 0;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_input(
        &self,
        runtime: &MetalRuntime,
        target_weights: &NativeWeights,
        target_model: &NativeModel,
        state: &mut MtpRequestState,
        token_id: u32,
        hidden: &[f32],
        need_logits: bool,
    ) -> Result<MtpForwardResult, NativeError> {
        if hidden.len() != self.config.hidden_size {
            return Err(NativeError::VectorLengthMismatch {
                actual: hidden.len(),
                expected: self.config.hidden_size,
            });
        }
        let embedding = dequantized_row(
            target_weights,
            &target_model.embed_tokens,
            token_id as usize,
        )?;
        let embedding = rms_norm(
            &embedding,
            &self.pre_fc_norm_embedding,
            self.config.rms_norm_eps,
        )?;
        let hidden = rms_norm(hidden, &self.pre_fc_norm_hidden, self.config.rms_norm_eps)?;
        let mut fc_input = Vec::with_capacity(embedding.len() + hidden.len());
        fc_input.extend_from_slice(&embedding);
        fc_input.extend_from_slice(&hidden);

        if std::env::var_os("QWEN38_DISABLE_MTP_GPU_DECODE").is_none() {
            let fc_jobs = self.weights.mapped_q4_jobs(&[&self.fc], fc_input.len())?;
            let rope = self.config.rope();
            let layer = self.layer.gpu_decode_layer(
                &self.weights,
                MropePosition::text(state.next_position),
                &rope,
            )?;
            let lm_jobs = need_logits
                .then(|| {
                    target_weights.mapped_q4_jobs(&[&target_model.lm_head], self.config.hidden_size)
                })
                .transpose()?;
            runtime
                .mtp_decode_step(
                    &mut state.decode,
                    &fc_input,
                    &fc_jobs[0],
                    &layer,
                    self.mtp_mlp_f16.as_ref(),
                    &mut state.kv,
                    &self.norm_gpu,
                    lm_jobs.as_ref().map(|jobs| &jobs[0]),
                    self.config.rms_norm_eps,
                )
                .map_err(NativeError::Metal)?;
            state.next_position = state.next_position.saturating_add(1);
            let hidden = runtime
                .read_decode_normalized(&state.decode)
                .map_err(NativeError::Metal)?;
            let token = lm_jobs
                .is_some()
                .then(|| runtime.read_decode_token(&state.decode))
                .transpose()
                .map_err(NativeError::Metal)?;
            return Ok(MtpForwardResult {
                hidden,
                logits: None,
                token,
            });
        }

        let projected = self
            .weights
            .q4_affine_matvec(runtime, &self.fc, &fc_input)?;
        let output = self.layer.forward(
            runtime,
            &self.weights,
            &projected,
            &mut state.kv,
            MropePosition::text(state.next_position),
            self.config.rope(),
            self.config.rms_norm_eps,
        )?;
        state.next_position = state.next_position.saturating_add(1);
        let mut hidden = projected;
        add_in_place(&mut hidden, &output)?;
        let post_norm = rms_norm(
            &hidden,
            &self.layer.common.post_attention_norm,
            self.config.rms_norm_eps,
        )?;
        let mlp = self.weights.q4_affine_mlp_batch(
            runtime,
            &self.layer.common.gate_proj,
            &self.layer.common.up_proj,
            &self.layer.common.down_proj,
            &post_norm,
            1,
        )?;
        add_in_place(&mut hidden, &mlp)?;
        let hidden = rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;
        let logits = if need_logits {
            Some(self.target_logits(runtime, target_weights, target_model, &hidden)?)
        } else {
            None
        };
        Ok(MtpForwardResult {
            hidden,
            logits,
            token: None,
        })
    }

    fn target_logits(
        &self,
        runtime: &MetalRuntime,
        target_weights: &NativeWeights,
        target_model: &NativeModel,
        hidden: &[f32],
    ) -> Result<Vec<f32>, NativeError> {
        target_weights.q4_affine_matvec(runtime, &target_model.lm_head, hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_block(
        &self,
        runtime: &MetalRuntime,
        target_weights: &NativeWeights,
        target_model: &NativeModel,
        state: &mut MtpRequestState,
        bonus: u32,
        target_hidden: &[f32],
        draft_count: usize,
    ) -> Result<Vec<u32>, NativeError> {
        let mut previous_hidden = target_hidden.to_vec();
        let mut current_token = bonus;
        let mut tokens = Vec::with_capacity(draft_count);
        let mut appended = 0;
        let mut seed_hidden_available = true;
        if let Some(seed_token) = state.seed_token.take() {
            current_token = seed_token;
            if let Some(seed_hidden) = state.seed_hidden.take() {
                previous_hidden = seed_hidden;
            } else {
                // The fused default path only needs the token itself because
                // it proposes one item. An experimental multi-token block
                // must retain the adapter activation to advance further.
                seed_hidden_available = false;
            }
            tokens.push(current_token);
        }
        while tokens.len() < draft_count {
            if !seed_hidden_available && !tokens.is_empty() {
                return Err(NativeError::InvalidConfig(
                    "MTP seed activation is required for a multi-token draft".to_owned(),
                ));
            }
            let output = self.forward_input(
                runtime,
                target_weights,
                target_model,
                state,
                current_token,
                &previous_hidden,
                true,
            )?;
            appended += 1;
            previous_hidden = output.hidden;
            seed_hidden_available = true;
            current_token = output
                .token
                .or_else(|| output.logits.as_ref().map(|logits| argmax(logits)))
                .ok_or_else(|| {
                    NativeError::InvalidConfig("MTP draft step did not produce logits".to_owned())
                })?;
            tokens.push(current_token);
        }
        state.round_appended = appended;
        Ok(tokens)
    }

    #[allow(clippy::too_many_arguments)]
    fn accept_verified(
        &self,
        runtime: &MetalRuntime,
        target_weights: &NativeWeights,
        target_model: &NativeModel,
        state: &mut MtpRequestState,
        draft_tokens: &[u32],
        accepted: usize,
        target_bonus: u32,
        verify_hidden: &[f32],
    ) -> Result<(), NativeError> {
        let hidden_size = self.config.hidden_size;
        let keep_appended = accepted.min(state.round_appended);
        let trim = state.round_appended.saturating_sub(keep_appended);
        if trim > 0 {
            let sequence_length =
                state
                    .kv
                    .sequence_length()
                    .checked_sub(trim)
                    .ok_or_else(|| {
                        NativeError::InvalidConfig("MTP KV rollback underflow".to_owned())
                    })?;
            runtime
                .truncate_q8_kv_tokens(&mut state.kv, sequence_length)
                .map_err(NativeError::Metal)?;
            state.next_position = state.next_position.saturating_sub(trim as u32);
        }

        for (index, &draft_token) in draft_tokens
            .iter()
            .enumerate()
            .skip(keep_appended)
            .take(accepted.saturating_sub(keep_appended))
        {
            let offset = index.checked_mul(hidden_size).ok_or_else(|| {
                NativeError::DimensionOverflow("MTP verify hidden offset".to_owned())
            })?;
            let end = offset.checked_add(hidden_size).ok_or_else(|| {
                NativeError::DimensionOverflow("MTP verify hidden end".to_owned())
            })?;
            let hidden = verify_hidden.get(offset..end).ok_or_else(|| {
                NativeError::VectorLengthMismatch {
                    actual: verify_hidden.len(),
                    expected: (accepted + 1) * hidden_size,
                }
            })?;
            let _ = self.forward_input(
                runtime,
                target_weights,
                target_model,
                state,
                draft_token,
                hidden,
                false,
            )?;
        }

        let offset = accepted
            .checked_mul(hidden_size)
            .ok_or_else(|| NativeError::DimensionOverflow("MTP bonus hidden offset".to_owned()))?;
        let end = offset
            .checked_add(hidden_size)
            .ok_or_else(|| NativeError::DimensionOverflow("MTP bonus hidden end".to_owned()))?;
        let hidden =
            verify_hidden
                .get(offset..end)
                .ok_or_else(|| NativeError::VectorLengthMismatch {
                    actual: verify_hidden.len(),
                    expected: (accepted + 1) * hidden_size,
                })?;
        let seed = self.forward_input(
            runtime,
            target_weights,
            target_model,
            state,
            target_bonus,
            hidden,
            true,
        )?;
        let seed_token = seed
            .token
            .or_else(|| seed.logits.as_ref().map(|logits| argmax(logits)))
            .ok_or_else(|| {
                NativeError::InvalidConfig("MTP seed step did not produce logits".to_owned())
            })?;
        state.seed_token = Some(seed_token);
        state.seed_hidden = Some(seed.hidden);
        state.round_appended = 0;
        Ok(())
    }
}

fn validate_mtp_pair(
    target: &TextRuntimeConfig,
    adapter: &TextRuntimeConfig,
) -> Result<(), NativeError> {
    for (name, target_value, adapter_value) in [
        ("hidden_size", target.hidden_size, adapter.hidden_size),
        ("vocab_size", target.vocab_size, adapter.vocab_size),
        (
            "num_attention_heads",
            target.num_attention_heads,
            adapter.num_attention_heads,
        ),
        (
            "num_key_value_heads",
            target.num_key_value_heads,
            adapter.num_key_value_heads,
        ),
        ("head_dim", target.head_dim, adapter.head_dim),
    ] {
        if target_value != adapter_value {
            return Err(NativeError::InvalidConfig(format!(
                "MTP adapter {name}={adapter_value} does not match target {target_value}"
            )));
        }
    }
    Ok(())
}

impl LayerWeights {
    fn input_norm(&self) -> &[f32] {
        match self {
            Self::Linear(layer) => &layer.common.input_norm,
            Self::Full(layer) => &layer.common.input_norm,
        }
    }

    fn post_attention_norm(&self) -> &[f32] {
        match self {
            Self::Linear(layer) => &layer.common.post_attention_norm,
            Self::Full(layer) => &layer.common.post_attention_norm,
        }
    }

    fn mlp_projections(&self) -> (&Q4AffineMatrix, &Q4AffineMatrix, &Q4AffineMatrix) {
        match self {
            Self::Linear(layer) => (
                &layer.common.gate_proj,
                &layer.common.up_proj,
                &layer.common.down_proj,
            ),
            Self::Full(layer) => (
                &layer.common.gate_proj,
                &layer.common.up_proj,
                &layer.common.down_proj,
            ),
        }
    }
}

fn load_vector(
    weights: &NativeWeights,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>, NativeError> {
    let values = weights.tensor_values_f32(name)?;
    if values.len() != expected {
        return Err(NativeError::WrongVectorLength {
            name: name.to_owned(),
            actual: values.len(),
            expected,
        });
    }
    Ok(values)
}

struct RuntimeState {
    layers: Vec<LayerRuntimeState>,
    decode: MetalDecodeState,
    /// Reusable short-batch activation state for MTP target verification.
    /// Its buffers are request-local scratch and do not participate in
    /// speculation commit/rollback.
    verify_batch: Option<MetalBatchDecodeState>,
    speculation: Option<SpeculationWorkspace>,
}

struct SpeculationWorkspace {
    /// One reusable destination state for each DeltaNet layer. The buffers are
    /// allocated only when a request first enters speculative verification.
    shadow_linear: Vec<Option<MetalDeltaNetState>>,
    /// Optional per-row state images used to commit a partially accepted
    /// verification block without replaying its target prefix.
    snapshots: Vec<Option<MetalDeltaNetSnapshots>>,
    /// Sequence lengths captured before the current verification transaction.
    /// Full-attention KV bytes can be appended in place and rolled back by
    /// restoring these logical lengths.
    full_lengths: Vec<usize>,
    active: bool,
}

enum LayerRuntimeState {
    Linear(MetalDeltaNetState),
    Full(Q8KvState),
}

impl RuntimeState {
    fn new(model: &NativeModel, runtime: &MetalRuntime) -> Result<Self, NativeError> {
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            match layer {
                LayerWeights::Linear(layer) => layers.push(LayerRuntimeState::Linear(
                    runtime
                        .create_deltanet_state(&layer.delta)
                        .map_err(NativeError::Metal)?,
                )),
                LayerWeights::Full(layer) => layers.push(LayerRuntimeState::Full(
                    runtime
                        .create_q8_kv_state(layer.num_key_value_heads, layer.head_dim)
                        .map_err(NativeError::Metal)?,
                )),
            }
        }
        Ok(Self {
            layers,
            decode: runtime
                .create_decode_state(model.config.hidden_size)
                .map_err(NativeError::Metal)?,
            verify_batch: None,
            speculation: None,
        })
    }

    fn reserve_prefill(
        &mut self,
        runtime: &MetalRuntime,
        token_count: usize,
    ) -> Result<(), NativeError> {
        for layer in &mut self.layers {
            if let LayerRuntimeState::Full(state) = layer {
                runtime
                    .reserve_q8_kv_tokens(state, token_count)
                    .map_err(NativeError::Metal)?;
            }
        }
        Ok(())
    }

    fn begin_speculation(
        &mut self,
        model: &NativeModel,
        runtime: &MetalRuntime,
    ) -> Result<(), NativeError> {
        if self.layers.len() != model.layers.len() {
            return Err(NativeError::InvalidConfig(
                "runtime state layer count does not match model".to_owned(),
            ));
        }
        let workspace = self
            .speculation
            .get_or_insert_with(|| SpeculationWorkspace {
                shadow_linear: (0..model.layers.len()).map(|_| None).collect(),
                snapshots: (0..model.layers.len()).map(|_| None).collect(),
                full_lengths: Vec::with_capacity(model.layers.len()),
                active: false,
            });
        if workspace.active {
            return Err(NativeError::InvalidConfig(
                "speculative state transaction is already active".to_owned(),
            ));
        }
        workspace.full_lengths.clear();
        workspace.full_lengths.resize(model.layers.len(), 0);
        for (layer_index, (layer, layer_state)) in
            model.layers.iter().zip(self.layers.iter()).enumerate()
        {
            match (layer, layer_state) {
                (LayerWeights::Linear(linear), LayerRuntimeState::Linear(_)) => {
                    if workspace.shadow_linear[layer_index].is_none() {
                        workspace.shadow_linear[layer_index] = Some(
                            runtime
                                .create_deltanet_state(&linear.delta)
                                .map_err(NativeError::Metal)?,
                        );
                    }
                }
                (LayerWeights::Full(_), LayerRuntimeState::Full(state)) => {
                    workspace.full_lengths[layer_index] = state.sequence_length();
                }
                _ => unreachable!("layer weights and runtime state are constructed together"),
            }
        }
        workspace.active = true;
        Ok(())
    }

    fn commit_speculation(&mut self, model: &NativeModel) -> Result<(), NativeError> {
        let Some(workspace) = self.speculation.as_mut() else {
            return Err(NativeError::InvalidConfig(
                "speculative state transaction is not active".to_owned(),
            ));
        };
        if !workspace.active {
            return Err(NativeError::InvalidConfig(
                "speculative state transaction is not active".to_owned(),
            ));
        }
        for (layer_index, (layer, layer_state)) in
            model.layers.iter().zip(self.layers.iter_mut()).enumerate()
        {
            if matches!(layer, LayerWeights::Linear(_)) {
                let LayerRuntimeState::Linear(active) = layer_state else {
                    unreachable!("layer weights and runtime state are constructed together")
                };
                let shadow = workspace.shadow_linear[layer_index]
                    .as_mut()
                    .ok_or_else(|| {
                        NativeError::InvalidConfig("missing DeltaNet speculation shadow".to_owned())
                    })?;
                // The old active buffers remain in `shadow` and can be reused
                // as the destination of the next transaction.
                self_runtime_swap_deltanet_state(active, shadow);
            }
        }
        workspace.active = false;
        Ok(())
    }

    fn prepare_speculation_snapshots(
        &mut self,
        model: &NativeModel,
        runtime: &MetalRuntime,
        row_count: usize,
    ) -> Result<(), NativeError> {
        if !mtp_state_snapshots_enabled() || row_count == 0 {
            return Ok(());
        }
        let workspace = self.speculation.as_mut().ok_or_else(|| {
            NativeError::InvalidConfig("speculative state transaction is not active".to_owned())
        })?;
        if !workspace.active {
            return Err(NativeError::InvalidConfig(
                "speculative state transaction is not active".to_owned(),
            ));
        }
        for (layer_index, (layer, layer_state)) in
            model.layers.iter().zip(self.layers.iter()).enumerate()
        {
            if let (LayerWeights::Linear(linear), LayerRuntimeState::Linear(_)) =
                (layer, layer_state)
            {
                let needs_allocation = workspace.snapshots[layer_index]
                    .as_ref()
                    .is_none_or(|snapshots| snapshots.row_count() < row_count);
                if needs_allocation {
                    workspace.snapshots[layer_index] = Some(
                        runtime
                            .create_deltanet_snapshots(&linear.delta, row_count)
                            .map_err(NativeError::Metal)?,
                    );
                }
            }
        }
        Ok(())
    }

    fn has_speculation_snapshots(&self, model: &NativeModel) -> bool {
        if !mtp_state_snapshots_enabled() {
            return false;
        }
        let Some(workspace) = self.speculation.as_ref() else {
            return false;
        };
        workspace.active
            && model
                .layers
                .iter()
                .enumerate()
                .all(|(layer_index, layer)| match layer {
                    LayerWeights::Linear(_) => workspace.snapshots[layer_index].is_some(),
                    LayerWeights::Full(_) => true,
                })
    }

    /// Commits the first `rows` rows of an active verification transaction.
    /// DeltaNet rows are restored from their GPU-produced snapshots and full
    /// attention simply shortens its logical KV length.
    fn commit_speculation_prefix(
        &mut self,
        model: &NativeModel,
        runtime: &MetalRuntime,
        rows: usize,
    ) -> Result<(), NativeError> {
        if rows == 0 {
            return Err(NativeError::InvalidConfig(
                "speculative prefix must contain at least one row".to_owned(),
            ));
        }
        let workspace = self.speculation.as_mut().ok_or_else(|| {
            NativeError::InvalidConfig("speculative state transaction is not active".to_owned())
        })?;
        if !workspace.active {
            return Err(NativeError::InvalidConfig(
                "speculative state transaction is not active".to_owned(),
            ));
        }
        let snapshot_row = rows - 1;
        for (layer_index, (layer, layer_state)) in
            model.layers.iter().zip(self.layers.iter_mut()).enumerate()
        {
            match (layer, layer_state) {
                (LayerWeights::Linear(_), LayerRuntimeState::Linear(active)) => {
                    let snapshots = workspace.snapshots[layer_index].as_mut().ok_or_else(|| {
                        NativeError::InvalidConfig(
                            "missing DeltaNet state snapshots for partial commit".to_owned(),
                        )
                    })?;
                    runtime
                        .restore_deltanet_snapshot(snapshots, snapshot_row, active)
                        .map_err(NativeError::Metal)?;
                }
                (LayerWeights::Full(_), LayerRuntimeState::Full(state)) => {
                    let sequence_length = workspace.full_lengths[layer_index]
                        .checked_add(rows)
                        .ok_or_else(|| {
                            NativeError::DimensionOverflow(
                                "speculative full-attention prefix length".to_owned(),
                            )
                        })?;
                    if sequence_length > state.sequence_length() {
                        return Err(NativeError::InvalidConfig(
                            "speculative full-attention state is shorter than the accepted prefix"
                                .to_owned(),
                        ));
                    }
                    runtime
                        .truncate_q8_kv_tokens(state, sequence_length)
                        .map_err(NativeError::Metal)?;
                }
                _ => unreachable!("layer weights and runtime state are constructed together"),
            }
        }
        workspace.active = false;
        Ok(())
    }

    fn rollback_speculation(
        &mut self,
        model: &NativeModel,
        runtime: &MetalRuntime,
    ) -> Result<(), NativeError> {
        let Some(workspace) = self.speculation.as_mut() else {
            return Err(NativeError::InvalidConfig(
                "speculative state transaction is not active".to_owned(),
            ));
        };
        if !workspace.active {
            return Err(NativeError::InvalidConfig(
                "speculative state transaction is not active".to_owned(),
            ));
        }
        for (layer_index, (layer, layer_state)) in
            model.layers.iter().zip(self.layers.iter_mut()).enumerate()
        {
            if let (LayerWeights::Full(_), LayerRuntimeState::Full(state)) = (layer, layer_state) {
                runtime
                    .truncate_q8_kv_tokens(state, workspace.full_lengths[layer_index])
                    .map_err(NativeError::Metal)?;
            }
        }
        workspace.active = false;
        Ok(())
    }

    fn fork(&self, model: &NativeModel, runtime: &MetalRuntime) -> Result<Self, NativeError> {
        if self.layers.len() != model.layers.len() {
            return Err(NativeError::InvalidConfig(
                "runtime state layer count does not match model".to_owned(),
            ));
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(match layer {
                LayerRuntimeState::Linear(state) => LayerRuntimeState::Linear(
                    runtime
                        .clone_deltanet_state(state)
                        .map_err(NativeError::Metal)?,
                ),
                LayerRuntimeState::Full(state) => LayerRuntimeState::Full(
                    runtime
                        .clone_q8_kv_state(state)
                        .map_err(NativeError::Metal)?,
                ),
            });
        }
        Ok(Self {
            layers,
            // Decode scratch is request-local and contains no model state that
            // needs to be copied. A fresh stream also prevents concurrent
            // requests from aliasing activation buffers.
            decode: runtime
                .create_decode_state(model.config.hidden_size)
                .map_err(NativeError::Metal)?,
            verify_batch: None,
            speculation: None,
        })
    }
}

fn self_runtime_swap_deltanet_state(
    active: &mut MetalDeltaNetState,
    shadow: &mut MetalDeltaNetState,
) {
    std::mem::swap(active, shadow);
}

impl LinearLayerWeights {
    fn gpu_decode_layer<'a>(
        &'a self,
        weights: &'a NativeWeights,
        state: &'a MetalDeltaNetState,
    ) -> Result<MetalDecodeLinearLayer<'a>, NativeError> {
        let hidden_elements = self.common.input_norm.len();
        let input_jobs = weights.mapped_q4_jobs(
            &[
                &self.in_proj_qkv,
                &self.in_proj_z,
                &self.in_proj_b,
                &self.in_proj_a,
            ],
            hidden_elements,
        )?;
        let delta_elements = usize::try_from(self.out_proj.input_elements)
            .map_err(|_| NativeError::DimensionOverflow("DeltaNet output elements".to_owned()))?;
        let out_jobs = weights.mapped_q4_jobs(&[&self.out_proj], delta_elements)?;
        let mlp_jobs = weights.mapped_q4_jobs(
            &[&self.common.gate_proj, &self.common.up_proj],
            hidden_elements,
        )?;
        let mlp_elements = usize::try_from(self.common.gate_proj.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("MLP gate output rows".to_owned()))?;
        let down_jobs = weights.mapped_q4_jobs(&[&self.common.down_proj], mlp_elements)?;
        Ok(MetalDecodeLinearLayer::new(
            &self.common.input_norm_gpu,
            &self.common.post_attention_norm_gpu,
            input_jobs[0],
            input_jobs[1],
            input_jobs[2],
            input_jobs[3],
            out_jobs[0],
            &self.delta,
            state,
            mlp_jobs[0],
            mlp_jobs[1],
            down_jobs[0],
        ))
    }

    fn forward(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        input: &[f32],
        state: &mut MetalDeltaNetState,
        eps: f32,
    ) -> Result<Vec<f32>, NativeError> {
        let mut projections = weights.q4_affine_matvec_batch(
            runtime,
            &[
                &self.in_proj_qkv,
                &self.in_proj_z,
                &self.in_proj_b,
                &self.in_proj_a,
            ],
            input,
        )?;
        let qkv = projections.remove(0);
        let z = projections.remove(0);
        let b = projections.remove(0);
        let a = projections.remove(0);
        let output = runtime
            .deltanet_step(&self.delta, state, &qkv, &z, &b, &a, eps)
            .map_err(NativeError::Metal)?;
        let projected = weights.q4_affine_matvec(runtime, &self.out_proj, &output)?;
        Ok(projected)
    }

    fn forward_prefill(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        input: &[f32],
        batch_size: usize,
        state: &mut MetalDeltaNetState,
        eps: f32,
    ) -> Result<Vec<f32>, NativeError> {
        let mut projections = weights.q4_affine_matmul_batch(
            runtime,
            &[
                &self.in_proj_qkv,
                &self.in_proj_z,
                &self.in_proj_b,
                &self.in_proj_a,
            ],
            input,
            batch_size,
        )?;
        let qkv = projections.remove(0);
        let z = projections.remove(0);
        let b = projections.remove(0);
        let a = projections.remove(0);
        let qkv_width = usize::try_from(self.in_proj_qkv.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("DeltaNet qkv rows".to_owned()))?;
        let z_width = usize::try_from(self.in_proj_z.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("DeltaNet z rows".to_owned()))?;
        let b_width = usize::try_from(self.in_proj_b.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("DeltaNet b rows".to_owned()))?;
        let a_width = usize::try_from(self.in_proj_a.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("DeltaNet a rows".to_owned()))?;
        ensure_batched_width(&qkv, batch_size, qkv_width, "DeltaNet qkv prefill")?;
        ensure_batched_width(&z, batch_size, z_width, "DeltaNet z prefill")?;
        ensure_batched_width(&b, batch_size, b_width, "DeltaNet b prefill")?;
        ensure_batched_width(&a, batch_size, a_width, "DeltaNet a prefill")?;

        let output = runtime
            .deltanet_prefill(&self.delta, state, &qkv, &z, &b, &a, batch_size, eps)
            .map_err(NativeError::Metal)?;
        let mut projected =
            weights.q4_affine_matmul_batch(runtime, &[&self.out_proj], &output, batch_size)?;
        Ok(projected.remove(0))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_prefill_from(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        input: &[f32],
        batch_size: usize,
        source: &MetalDeltaNetState,
        destination: &mut MetalDeltaNetState,
        eps: f32,
    ) -> Result<Vec<f32>, NativeError> {
        let mut projections = weights.q4_affine_matmul_batch(
            runtime,
            &[
                &self.in_proj_qkv,
                &self.in_proj_z,
                &self.in_proj_b,
                &self.in_proj_a,
            ],
            input,
            batch_size,
        )?;
        let qkv = projections.remove(0);
        let z = projections.remove(0);
        let b = projections.remove(0);
        let a = projections.remove(0);
        let qkv_width = usize::try_from(self.in_proj_qkv.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("DeltaNet qkv rows".to_owned()))?;
        let z_width = usize::try_from(self.in_proj_z.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("DeltaNet z rows".to_owned()))?;
        let b_width = usize::try_from(self.in_proj_b.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("DeltaNet b rows".to_owned()))?;
        let a_width = usize::try_from(self.in_proj_a.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("DeltaNet a rows".to_owned()))?;
        ensure_batched_width(&qkv, batch_size, qkv_width, "DeltaNet qkv prefill")?;
        ensure_batched_width(&z, batch_size, z_width, "DeltaNet z prefill")?;
        ensure_batched_width(&b, batch_size, b_width, "DeltaNet b prefill")?;
        ensure_batched_width(&a, batch_size, a_width, "DeltaNet a prefill")?;

        let output = runtime
            .deltanet_prefill_from(
                &self.delta,
                source,
                destination,
                &qkv,
                &z,
                &b,
                &a,
                batch_size,
                eps,
            )
            .map_err(NativeError::Metal)?;
        let mut projected =
            weights.q4_affine_matmul_batch(runtime, &[&self.out_proj], &output, batch_size)?;
        Ok(projected.remove(0))
    }
}

impl FullLayerWeights {
    fn gpu_decode_layer<'a>(
        &'a self,
        weights: &'a NativeWeights,
        position: MropePosition,
        rope: &'a RopeParameters,
    ) -> Result<MetalDecodeFullLayer<'a>, NativeError> {
        let hidden_elements = self.common.input_norm.len();
        let attention_jobs =
            weights.mapped_q4_jobs(&[&self.q_proj, &self.k_proj, &self.v_proj], hidden_elements)?;
        let query_elements = self
            .num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| {
                NativeError::DimensionOverflow("full-attention query elements".to_owned())
            })?;
        let output_jobs = weights.mapped_q4_jobs(&[&self.o_proj], query_elements)?;
        let mlp_jobs = weights.mapped_q4_jobs(
            &[&self.common.gate_proj, &self.common.up_proj],
            hidden_elements,
        )?;
        let mlp_elements = usize::try_from(self.common.gate_proj.output_rows).map_err(|_| {
            NativeError::DimensionOverflow("full-attention MLP elements".to_owned())
        })?;
        let down_jobs = weights.mapped_q4_jobs(&[&self.common.down_proj], mlp_elements)?;
        let (section1, section2, has_mrope_sections) = match rope.mrope_section.as_deref() {
            Some([_, section1, section2]) => (
                u32::try_from(*section1)
                    .map_err(|_| NativeError::DimensionOverflow("M-RoPE section one".to_owned()))?,
                u32::try_from(*section2)
                    .map_err(|_| NativeError::DimensionOverflow("M-RoPE section two".to_owned()))?,
                true,
            ),
            _ => (0, 0, false),
        };
        let rotary_dim = ((self.head_dim as f32 * rope.partial_rotary_factor).round() as usize)
            .min(self.head_dim);
        Ok(MetalDecodeFullLayer::new(
            &self.common.input_norm_gpu,
            &self.common.post_attention_norm_gpu,
            attention_jobs[0],
            attention_jobs[1],
            attention_jobs[2],
            output_jobs[0],
            &self.q_norm_gpu,
            &self.k_norm_gpu,
            MetalGqaDecodeConfig {
                num_heads: self.num_attention_heads,
                kv_heads: self.num_key_value_heads,
                head_dim: self.head_dim,
                rotary_dim,
                position: position.0,
                section1,
                section2,
                has_mrope_sections,
                rope_theta: rope.rope_theta,
            },
            mlp_jobs[0],
            mlp_jobs[1],
            down_jobs[0],
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        input: &[f32],
        state: &mut Q8KvState,
        position: MropePosition,
        rope: RopeParameters,
        eps: f32,
    ) -> Result<Vec<f32>, NativeError> {
        let mut projections = weights.q4_affine_matvec_batch(
            runtime,
            &[&self.q_proj, &self.k_proj, &self.v_proj],
            input,
        )?;
        let q_with_gate = projections.remove(0);
        let (mut query, gate) =
            split_query_and_gate(&q_with_gate, self.num_attention_heads, self.head_dim)?;
        let mut key = projections.remove(0);
        let value = projections.remove(0);
        let num_heads = self.num_attention_heads;
        let kv_heads = self.num_key_value_heads;
        let head_dim = self.head_dim;
        let expected_key_values = kv_heads.checked_mul(head_dim).ok_or_else(|| {
            NativeError::DimensionOverflow("full attention KV dimensions".to_owned())
        })?;
        if key.len() != expected_key_values || value.len() != expected_key_values {
            return Err(NativeError::VectorLengthMismatch {
                actual: key.len().min(value.len()),
                expected: expected_key_values,
            });
        }
        let rotary_dim =
            ((head_dim as f32 * rope.partial_rotary_factor).round() as usize).min(head_dim);
        for head in 0..num_heads {
            let offset = head * head_dim;
            let normalized = rms_norm_slice(&query[offset..offset + head_dim], &self.q_norm, eps);
            query[offset..offset + head_dim].copy_from_slice(&normalized);
            apply_mrope(
                &mut query[offset..offset + head_dim],
                position,
                rotary_dim,
                &rope,
            );
        }
        for head in 0..kv_heads {
            let offset = head * head_dim;
            let normalized = rms_norm_slice(&key[offset..offset + head_dim], &self.k_norm, eps);
            key[offset..offset + head_dim].copy_from_slice(&normalized);
            apply_mrope(
                &mut key[offset..offset + head_dim],
                position,
                rotary_dim,
                &rope,
            );
        }

        let attention_output = runtime
            .gqa_attention_q8(state, &query, &gate, &key, &value, num_heads)
            .map_err(NativeError::Metal)?;
        weights.q4_affine_matvec(runtime, &self.o_proj, &attention_output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_prefill(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        input: &[f32],
        positions: &[MropePosition],
        state: &mut Q8KvState,
        rope: &RopeParameters,
        eps: f32,
    ) -> Result<Vec<f32>, NativeError> {
        let batch_size = positions.len();
        let mut projections = weights.q4_affine_matmul_batch(
            runtime,
            &[&self.q_proj, &self.k_proj, &self.v_proj],
            input,
            batch_size,
        )?;
        let q_with_gate = projections.remove(0);
        let key = projections.remove(0);
        let value = projections.remove(0);
        let q_width = usize::try_from(self.q_proj.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("GQA q rows".to_owned()))?;
        let key_width = usize::try_from(self.k_proj.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("GQA k rows".to_owned()))?;
        let value_width = usize::try_from(self.v_proj.output_rows)
            .map_err(|_| NativeError::DimensionOverflow("GQA v rows".to_owned()))?;
        ensure_batched_width(&q_with_gate, batch_size, q_width, "GQA q prefill")?;
        ensure_batched_width(&key, batch_size, key_width, "GQA k prefill")?;
        ensure_batched_width(&value, batch_size, value_width, "GQA v prefill")?;

        let num_heads = self.num_attention_heads;
        let kv_heads = self.num_key_value_heads;
        let head_dim = self.head_dim;
        let expected_key_values = kv_heads.checked_mul(head_dim).ok_or_else(|| {
            NativeError::DimensionOverflow("full attention KV dimensions".to_owned())
        })?;
        if key_width != expected_key_values || value_width != expected_key_values {
            return Err(NativeError::VectorLengthMismatch {
                actual: key_width.min(value_width),
                expected: expected_key_values,
            });
        }
        let rotary_dim =
            ((head_dim as f32 * rope.partial_rotary_factor).round() as usize).min(head_dim);
        let output_width = num_heads.checked_mul(head_dim).ok_or_else(|| {
            NativeError::DimensionOverflow("full attention output dimensions".to_owned())
        })?;
        let mut queries = Vec::with_capacity(
            batch_size
                .checked_mul(output_width)
                .ok_or_else(|| NativeError::DimensionOverflow("GQA prefill query".to_owned()))?,
        );
        let mut gates = Vec::with_capacity(
            batch_size
                .checked_mul(output_width)
                .ok_or_else(|| NativeError::DimensionOverflow("GQA prefill gate".to_owned()))?,
        );
        let mut keys = Vec::with_capacity(
            batch_size
                .checked_mul(expected_key_values)
                .ok_or_else(|| NativeError::DimensionOverflow("GQA prefill key".to_owned()))?,
        );

        for ((q_with_gate, key), position) in q_with_gate
            .chunks_exact(q_width)
            .zip(key.chunks_exact(key_width))
            .zip(positions.iter().copied())
        {
            let (mut query, gate) = split_query_and_gate(q_with_gate, num_heads, head_dim)?;
            let mut key = key.to_vec();
            for head in 0..num_heads {
                let offset = head * head_dim;
                let normalized =
                    rms_norm_slice(&query[offset..offset + head_dim], &self.q_norm, eps);
                query[offset..offset + head_dim].copy_from_slice(&normalized);
                apply_mrope(
                    &mut query[offset..offset + head_dim],
                    position,
                    rotary_dim,
                    rope,
                );
            }
            for head in 0..kv_heads {
                let offset = head * head_dim;
                let normalized = rms_norm_slice(&key[offset..offset + head_dim], &self.k_norm, eps);
                key[offset..offset + head_dim].copy_from_slice(&normalized);
                apply_mrope(
                    &mut key[offset..offset + head_dim],
                    position,
                    rotary_dim,
                    rope,
                );
            }
            queries.extend_from_slice(&query);
            gates.extend_from_slice(&gate);
            keys.extend_from_slice(&key);
        }
        let attention_output = runtime
            .gqa_attention_q8_prefill(
                state, &queries, &gates, &keys, &value, num_heads, batch_size,
            )
            .map_err(NativeError::Metal)?;
        let mut projected = weights.q4_affine_matmul_batch(
            runtime,
            &[&self.o_proj],
            &attention_output,
            batch_size,
        )?;
        Ok(projected.remove(0))
    }
}

fn split_query_and_gate(
    values: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), NativeError> {
    let per_head = head_dim
        .checked_mul(2)
        .ok_or_else(|| NativeError::DimensionOverflow("attention head dimensions".to_owned()))?;
    let expected = num_heads.checked_mul(per_head).ok_or_else(|| {
        NativeError::DimensionOverflow("attention projection dimensions".to_owned())
    })?;
    if values.len() != expected {
        return Err(NativeError::VectorLengthMismatch {
            actual: values.len(),
            expected,
        });
    }

    let mut query = Vec::with_capacity(num_heads * head_dim);
    let mut gate = Vec::with_capacity(num_heads * head_dim);
    for head in values.chunks_exact(per_head) {
        query.extend_from_slice(&head[..head_dim]);
        gate.extend_from_slice(&head[head_dim..]);
    }
    Ok((query, gate))
}

fn dequantized_row(
    weights: &NativeWeights,
    matrix: &Q4AffineMatrix,
    row: usize,
) -> Result<Vec<f32>, NativeError> {
    if row >= matrix.output_rows as usize {
        return Err(NativeError::TokenOutOfRange(row as u32));
    }
    let packed = weights
        .store
        .tensor_data(matrix.weight_name())
        .ok_or_else(|| NativeError::MissingTensor(matrix.weight_name().to_owned()))?;
    let scales = weights
        .store
        .tensor_data(matrix.scales_name())
        .ok_or_else(|| NativeError::MissingTensor(matrix.scales_name().to_owned()))?;
    let biases = weights
        .store
        .tensor_data(matrix.biases_name())
        .ok_or_else(|| NativeError::MissingTensor(matrix.biases_name().to_owned()))?;
    let packed_columns = matrix.input_elements as usize / 8;
    let groups_per_row = matrix.input_elements as usize / AFFINE_GROUP_SIZE as usize;
    let mut output = vec![0.0; matrix.input_elements as usize];
    let packed_offset = row
        .checked_mul(packed_columns)
        .ok_or_else(|| NativeError::DimensionOverflow(matrix.weight_name().to_owned()))?;
    let affine_offset = row
        .checked_mul(groups_per_row)
        .ok_or_else(|| NativeError::DimensionOverflow(matrix.weight_name().to_owned()))?;
    for index in 0..output.len() {
        let word = u32::from_le_bytes([
            packed[(packed_offset + index / 8) * 4],
            packed[(packed_offset + index / 8) * 4 + 1],
            packed[(packed_offset + index / 8) * 4 + 2],
            packed[(packed_offset + index / 8) * 4 + 3],
        ]);
        let quantized = ((word >> ((index % 8) * 4)) & 0xF) as f32;
        let group = affine_offset + index / AFFINE_GROUP_SIZE as usize;
        let scale = bf16_bytes_to_f32(scales, group)?;
        let bias = bf16_bytes_to_f32(biases, group)?;
        output[index] = quantized * scale + bias;
    }
    Ok(output)
}

fn bf16_bytes_to_f32(bytes: &[u8], index: usize) -> Result<f32, NativeError> {
    let offset = index
        .checked_mul(2)
        .ok_or_else(|| NativeError::DimensionOverflow("BF16 parameter index".to_owned()))?;
    let value = bytes
        .get(offset..offset + 2)
        .ok_or(NativeError::InvalidParameterBytes)?;
    Ok(bf16_to_f32(u16::from_le_bytes([value[0], value[1]])))
}

fn validate_runtime_config(config: &TextRuntimeConfig) -> Result<(), NativeError> {
    let required = [
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("num_attention_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("max_position_embeddings", config.max_position_embeddings),
        ("vocab_size", config.vocab_size),
        ("linear_num_value_heads", config.linear_num_value_heads),
        ("linear_num_key_heads", config.linear_num_key_heads),
        ("linear_key_head_dim", config.linear_key_head_dim),
        ("linear_value_head_dim", config.linear_value_head_dim),
        ("linear_conv_kernel_dim", config.linear_conv_kernel_dim),
        ("head_dim", config.head_dim),
    ];
    if let Some((name, _)) = required.into_iter().find(|(_, value)| *value == 0) {
        return Err(NativeError::InvalidConfig(format!(
            "{name} must be greater than zero"
        )));
    }
    if config.num_attention_heads % config.num_key_value_heads != 0 {
        return Err(NativeError::InvalidConfig(
            "num_attention_heads must be divisible by num_key_value_heads".to_owned(),
        ));
    }
    if config.linear_num_value_heads % config.linear_num_key_heads != 0 {
        return Err(NativeError::InvalidConfig(
            "linear_num_value_heads must be divisible by linear_num_key_heads".to_owned(),
        ));
    }
    if config.layer_types.len() != config.num_hidden_layers {
        return Err(NativeError::InvalidConfig(format!(
            "layer_types has {} entries, expected {}",
            config.layer_types.len(),
            config.num_hidden_layers
        )));
    }
    Ok(())
}

fn validate_vision_runtime_config(config: &VisionRuntimeConfig) -> Result<(), NativeError> {
    for (name, value) in [
        ("vision.depth", config.depth),
        ("vision.hidden_size", config.hidden_size),
        ("vision.intermediate_size", config.intermediate_size),
        ("vision.num_heads", config.num_heads),
        (
            "vision.num_position_embeddings",
            config.num_position_embeddings,
        ),
        ("vision.out_hidden_size", config.out_hidden_size),
        ("vision.patch_size", config.patch_size),
        ("vision.temporal_patch_size", config.temporal_patch_size),
        ("vision.spatial_merge_size", config.spatial_merge_size),
        ("vision.in_channels", config.in_channels),
    ] {
        if value == 0 {
            return Err(NativeError::InvalidConfig(format!(
                "{name} must be greater than zero"
            )));
        }
    }
    if config.hidden_size % config.num_heads != 0 {
        return Err(NativeError::InvalidConfig(
            "vision.hidden_size must be divisible by vision.num_heads".to_owned(),
        ));
    }
    let side = (config.num_position_embeddings as f64).sqrt() as usize;
    if side * side != config.num_position_embeddings {
        return Err(NativeError::InvalidConfig(
            "vision.num_position_embeddings must describe a square embedding table".to_owned(),
        ));
    }
    Ok(())
}

fn add_in_place(destination: &mut [f32], source: &[f32]) -> Result<(), NativeError> {
    if destination.len() != source.len() {
        return Err(NativeError::VectorLengthMismatch {
            actual: source.len(),
            expected: destination.len(),
        });
    }
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += *source;
    }
    Ok(())
}

fn ensure_batched_width(
    values: &[f32],
    batch_size: usize,
    width: usize,
    name: &str,
) -> Result<(), NativeError> {
    let expected = batch_size
        .checked_mul(width)
        .ok_or_else(|| NativeError::DimensionOverflow(name.to_owned()))?;
    if values.len() != expected {
        return Err(NativeError::VectorLengthMismatch {
            actual: values.len(),
            expected,
        });
    }
    Ok(())
}

fn configured_prefill_chunk_tokens(batch_size: usize) -> Result<usize, NativeError> {
    let Some(value) = std::env::var_os("QWEN38_PREFILL_CHUNK_TOKENS") else {
        return Ok(batch_size.min(PREFILL_CHUNK_TOKENS));
    };
    let text = value.to_str().ok_or_else(|| {
        NativeError::InvalidConfig(
            "QWEN38_PREFILL_CHUNK_TOKENS must be a positive integer".to_owned(),
        )
    })?;
    let chunk_size = text.parse::<usize>().map_err(|_| {
        NativeError::InvalidConfig(
            "QWEN38_PREFILL_CHUNK_TOKENS must be a positive integer".to_owned(),
        )
    })?;
    if chunk_size == 0 {
        return Err(NativeError::InvalidConfig(
            "QWEN38_PREFILL_CHUNK_TOKENS must be a positive integer".to_owned(),
        ));
    }
    Ok(batch_size.min(chunk_size))
}

fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, NativeError> {
    if input.len() != weight.len() {
        return Err(NativeError::VectorLengthMismatch {
            actual: weight.len(),
            expected: input.len(),
        });
    }
    Ok(rms_norm_slice(input, weight, eps))
}

fn rms_norm_rows(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, NativeError> {
    if weight.is_empty() || input.len() % weight.len() != 0 {
        return Err(NativeError::VectorLengthMismatch {
            actual: input.len(),
            expected: weight.len(),
        });
    }
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(weight.len()) {
        output.extend(rms_norm_slice(row, weight, eps));
    }
    Ok(output)
}

fn rms_norm_slice(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean + eps).sqrt().recip();
    input
        .iter()
        .zip(weight)
        .map(|(input, weight)| input * scale * weight)
        .collect()
}

fn apply_mrope(
    values: &mut [f32],
    position: MropePosition,
    rotary_dim: usize,
    rope: &RopeParameters,
) {
    let half = rotary_dim / 2;
    for index in 0..half {
        let exponent = (2 * index) as f32 / rotary_dim as f32;
        let axis_position = position.axis_for_frequency(index, rope.mrope_section.as_deref());
        let angle = axis_position as f32 / rope.rope_theta.powf(exponent);
        let (sin, cos) = angle.sin_cos();
        let left = values[index];
        let right = values[index + half];
        values[index] = left * cos - right * sin;
        values[index + half] = right * cos + left * sin;
    }
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn mtp_request_is_eligible(request: &GenerationRequest, text_only: bool) -> bool {
    text_only
        && request.tools.is_empty()
        && !request.thinking.enabled
        && request.max_tokens >= 2
        && request.temperature.unwrap_or(0.0) <= f32::EPSILON
        && request.top_p.unwrap_or(1.0) >= 1.0 - f32::EPSILON
}

/// Returns the configured speculative depth. Apple Q4 verification is
/// bandwidth-bound, and one proposal keeps the target batch at two rows. The
/// adapter's wider trained block remains available as an explicit override.
fn configured_mtp_draft_limit(advertised: usize) -> Result<usize, NativeError> {
    let env_value = std::env::var_os("QWEN38_MTP_MAX_DRAFT_TOKENS");
    let raw = env_value
        .as_deref()
        .map(|value| {
            value.to_str().ok_or_else(|| {
                NativeError::InvalidConfig(
                    "QWEN38_MTP_MAX_DRAFT_TOKENS must be an ASCII integer".to_owned(),
                )
            })
        })
        .transpose()?;
    configured_mtp_draft_limit_with_override(advertised, raw)
}

fn configured_mtp_draft_limit_with_override(
    advertised: usize,
    raw: Option<&str>,
) -> Result<usize, NativeError> {
    const MAX_EXPERIMENTAL_DEPTH: usize = 8;
    let Some(raw) = raw else {
        return Ok(advertised.min(DEFAULT_MTP_DRAFT_TOKENS));
    };
    let depth = raw.parse::<usize>().map_err(|_| {
        NativeError::InvalidConfig(format!(
            "QWEN38_MTP_MAX_DRAFT_TOKENS must be between 1 and {MAX_EXPERIMENTAL_DEPTH}, got {raw:?}"
        ))
    })?;
    if !(1..=MAX_EXPERIMENTAL_DEPTH).contains(&depth) {
        return Err(NativeError::InvalidConfig(format!(
            "QWEN38_MTP_MAX_DRAFT_TOKENS must be between 1 and {MAX_EXPERIMENTAL_DEPTH}, got {depth}"
        )));
    }
    Ok(depth)
}

fn mtp_state_snapshots_enabled() -> bool {
    std::env::var_os("QWEN38_DISABLE_MTP_STATE_SNAPSHOTS").is_none()
}

fn sample_token(logits: &[f32], temperature: Option<f32>, top_p: Option<f32>, seed: u64) -> u32 {
    let temperature = temperature.unwrap_or(0.0);
    if temperature <= f32::EPSILON {
        return argmax(logits);
    }

    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(index, value)| (index, *value / temperature))
        .collect();
    let max_logit = candidates
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut normalizer = 0.0;
    for (_, value) in &mut candidates {
        *value = (*value - max_logit).exp();
        normalizer += *value;
    }
    for (_, value) in &mut candidates {
        *value /= normalizer.max(f32::MIN_POSITIVE);
    }
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    let limit = top_p.unwrap_or(1.0).clamp(f32::MIN_POSITIVE, 1.0);
    let mut cumulative = 0.0;
    let mut count = candidates.len();
    for (index, (_, probability)) in candidates.iter().enumerate() {
        cumulative += *probability;
        if cumulative >= limit {
            count = index + 1;
            break;
        }
    }
    let random_bits = splitmix64(seed);
    let total: f32 = candidates[..count].iter().map(|(_, value)| *value).sum();
    let mut random = (random_bits as f32 / u64::MAX as f32) * total;
    for (token, probability) in candidates.into_iter().take(count) {
        if random <= probability {
            return token as u32;
        }
        random -= probability;
    }
    argmax(logits)
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index as u32)
        .unwrap_or(0)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn token_sequence_stop_index(
    tokenizer: &Tokenizer,
    ids: &[u32],
    stop_sequences: &[String],
) -> Result<Option<usize>, NativeError> {
    if stop_sequences.is_empty() || ids.is_empty() {
        return Ok(None);
    }
    let text = tokenizer
        .decode(ids, false)
        .map_err(|error| NativeError::Tokenizer(error.to_string()))?;
    let Some(stop_index) = first_stop_index(&text, stop_sequences) else {
        return Ok(None);
    };
    let prefix = &text[..stop_index];
    let prefix_ids = tokenizer
        .encode(prefix, false)
        .map_err(|error| NativeError::Tokenizer(error.to_string()))?;
    Ok(Some(prefix_ids.len().min(ids.len())))
}

fn first_stop_index(text: &str, stop_sequences: &[String]) -> Option<usize> {
    stop_sequences
        .iter()
        .filter(|sequence| !sequence.is_empty())
        .filter_map(|sequence| text.find(sequence))
        .min()
}

/// A row-major BF16 tensor used by the visual encoder. MLX stores its dense
/// linear and Conv3d projection weights in this layout, including the patch
/// projection whose trailing dimensions are flattened into input columns.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Bf16Matrix {
    weight: MlxTensor,
    name: String,
    input_columns: usize,
    output_rows: usize,
}

impl Bf16Matrix {
    fn from_store(store: &MlxWeightStore, tensor_name: &str) -> Result<Self, NativeError> {
        let weight = store
            .tensor(tensor_name)
            .ok_or_else(|| NativeError::MissingTensor(tensor_name.to_owned()))?
            .clone();
        if weight.dtype != "BF16" || weight.shape.len() < 2 {
            return Err(NativeError::InvalidDenseWeight {
                name: tensor_name.to_owned(),
                dtype: weight.dtype,
                shape: weight.shape,
            });
        }
        let output_rows = usize::try_from(weight.shape[0])
            .map_err(|_| NativeError::DimensionOverflow(tensor_name.to_owned()))?;
        let input_columns = weight.shape[1..].iter().try_fold(1_usize, |total, value| {
            let value = usize::try_from(*value)
                .map_err(|_| NativeError::DimensionOverflow(tensor_name.to_owned()))?;
            total
                .checked_mul(value)
                .ok_or_else(|| NativeError::DimensionOverflow(tensor_name.to_owned()))
        })?;
        let expected_bytes = output_rows
            .checked_mul(input_columns)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| NativeError::DimensionOverflow(tensor_name.to_owned()))?;
        if output_rows == 0 || input_columns == 0 || weight.byte_len != expected_bytes as u64 {
            return Err(NativeError::InvalidDenseWeight {
                name: tensor_name.to_owned(),
                dtype: weight.dtype,
                shape: weight.shape,
            });
        }
        Ok(Self {
            weight,
            name: tensor_name.to_owned(),
            input_columns,
            output_rows,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q4AffineMatrix {
    weight: MlxTensor,
    scales: MlxTensor,
    biases: MlxTensor,
    weight_name: String,
    scales_name: String,
    biases_name: String,
    pub input_elements: u64,
    pub output_rows: u64,
}

impl Q4AffineMatrix {
    fn from_store(store: &MlxWeightStore, tensor_name: &str) -> Result<Self, NativeError> {
        let weight = store
            .tensor(tensor_name)
            .ok_or_else(|| NativeError::MissingTensor(tensor_name.to_owned()))?
            .clone();
        let base = tensor_name
            .strip_suffix(".weight")
            .ok_or_else(|| NativeError::NotQuantizedWeight(tensor_name.to_owned()))?;
        let scales_name = format!("{base}.scales");
        let biases_name = format!("{base}.biases");
        let scales = store
            .tensor(&scales_name)
            .ok_or(NativeError::MissingTensor(scales_name))?
            .clone();
        let biases = store
            .tensor(&biases_name)
            .ok_or(NativeError::MissingTensor(biases_name))?
            .clone();
        Self::from_tensors(tensor_name, weight, scales, biases)
    }

    fn from_tensors(
        name: &str,
        weight: MlxTensor,
        scales: MlxTensor,
        biases: MlxTensor,
    ) -> Result<Self, NativeError> {
        if weight.dtype != "U32" || weight.shape.len() != 2 {
            return Err(NativeError::InvalidQuantizedWeight {
                name: name.to_owned(),
                dtype: weight.dtype,
                shape: weight.shape,
            });
        }
        let output_rows = weight.shape[0];
        let packed_columns = weight.shape[1];
        let input_elements = packed_columns
            .checked_mul(VALUES_PER_PACKED_WORD)
            .ok_or(NativeError::DimensionOverflow(name.to_owned()))?;
        if output_rows == 0 || input_elements == 0 || input_elements % AFFINE_GROUP_SIZE != 0 {
            return Err(NativeError::InvalidQuantizedWeight {
                name: name.to_owned(),
                dtype: weight.dtype,
                shape: weight.shape,
            });
        }
        let expected_weight_bytes = output_rows
            .checked_mul(packed_columns)
            .and_then(|values| values.checked_mul(4))
            .ok_or(NativeError::DimensionOverflow(name.to_owned()))?;
        if weight.byte_len != expected_weight_bytes {
            return Err(NativeError::TensorByteLength {
                name: name.to_owned(),
                actual: weight.byte_len,
                expected: expected_weight_bytes,
            });
        }

        let groups_per_row = input_elements / AFFINE_GROUP_SIZE;
        validate_affine_tensor(name, "scales", &scales, output_rows, groups_per_row)?;
        validate_affine_tensor(name, "biases", &biases, output_rows, groups_per_row)?;
        let base = name
            .strip_suffix(".weight")
            .expect("quantized matrix names are validated by the caller");

        Ok(Self {
            weight,
            scales,
            biases,
            weight_name: name.to_owned(),
            scales_name: format!("{base}.scales"),
            biases_name: format!("{base}.biases"),
            input_elements,
            output_rows,
        })
    }

    fn weight_name(&self) -> &str {
        &self.weight_name
    }

    fn scales_name(&self) -> &str {
        &self.scales_name
    }

    fn biases_name(&self) -> &str {
        &self.biases_name
    }
}

fn validate_affine_tensor(
    weight_name: &str,
    label: &'static str,
    tensor: &MlxTensor,
    output_rows: u64,
    groups_per_row: u64,
) -> Result<(), NativeError> {
    let expected_shape = vec![output_rows, groups_per_row];
    if tensor.dtype != "BF16" || tensor.shape != expected_shape {
        return Err(NativeError::InvalidAffineTensor {
            weight: weight_name.to_owned(),
            label,
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
            expected_shape,
        });
    }
    let expected_bytes = output_rows
        .checked_mul(groups_per_row)
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| NativeError::DimensionOverflow(weight_name.to_owned()))?;
    if tensor.byte_len != expected_bytes {
        return Err(NativeError::TensorByteLength {
            name: format!("{weight_name}.{label}"),
            actual: tensor.byte_len,
            expected: expected_bytes,
        });
    }
    Ok(())
}

#[derive(Debug)]
pub enum NativeError {
    Model(crate::model::ModelFormatError),
    Metal(MetalRuntimeError),
    MissingTensor(String),
    NotQuantizedWeight(String),
    InvalidQuantizedWeight {
        name: String,
        dtype: String,
        shape: Vec<u64>,
    },
    InvalidDenseWeight {
        name: String,
        dtype: String,
        shape: Vec<u64>,
    },
    InvalidAffineTensor {
        weight: String,
        label: &'static str,
        dtype: String,
        shape: Vec<u64>,
        expected_shape: Vec<u64>,
    },
    TensorByteLength {
        name: String,
        actual: u64,
        expected: u64,
    },
    InputDimension {
        actual: usize,
        expected: u64,
    },
    MissingMappedShard(usize),
    DimensionOverflow(String),
    ConfigRead {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    ConfigJson(serde_json::Error),
    GenerationConfigJson(serde_json::Error),
    MissingTextConfig,
    Tokenizer(String),
    Prompt(String),
    EmptyPrompt,
    InvalidConfig(String),
    WrongVectorLength {
        name: String,
        actual: usize,
        expected: usize,
    },
    VectorLengthMismatch {
        actual: usize,
        expected: usize,
    },
    InvalidParameterBytes,
    Image(String),
    TokenOutOfRange(u32),
    ContextLimit {
        requested: u32,
        maximum: u32,
    },
    Unavailable(String),
    Streaming(String),
    Preflight(crate::preflight::PreflightError),
    PrefixCachePoisoned,
    MtpControllerPoisoned,
}

impl fmt::Display for NativeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(formatter, "invalid MLX model: {error}"),
            Self::Metal(error) => write!(formatter, "Metal execution failed: {error}"),
            Self::MissingTensor(name) => write!(formatter, "required tensor {name:?} is absent"),
            Self::NotQuantizedWeight(name) => {
                write!(
                    formatter,
                    "tensor {name:?} is not a quantized .weight tensor"
                )
            }
            Self::InvalidQuantizedWeight { name, dtype, shape } => write!(
                formatter,
                "quantized tensor {name:?} must be U32 with two dimensions, got {dtype} {shape:?}"
            ),
            Self::InvalidDenseWeight { name, dtype, shape } => write!(
                formatter,
                "dense tensor {name:?} must be BF16 with at least two dimensions, got {dtype} {shape:?}"
            ),
            Self::InvalidAffineTensor {
                weight,
                label,
                dtype,
                shape,
                expected_shape,
            } => write!(
                formatter,
                "{label} for {weight:?} must be BF16 {expected_shape:?}, got {dtype} {shape:?}"
            ),
            Self::TensorByteLength {
                name,
                actual,
                expected,
            } => write!(
                formatter,
                "tensor {name:?} occupies {actual} bytes, expected {expected}"
            ),
            Self::InputDimension { actual, expected } => write!(
                formatter,
                "input has {actual} elements, but the Q4 matrix expects {expected}"
            ),
            Self::MissingMappedShard(index) => write!(formatter, "mapped shard {index} is absent"),
            Self::DimensionOverflow(name) => {
                write!(formatter, "dimensions for {name:?} overflow u64")
            }
            Self::ConfigRead { path, source } => {
                write!(formatter, "cannot read {}: {source}", path.display())
            }
            Self::ConfigJson(error) => write!(formatter, "cannot parse config.json: {error}"),
            Self::GenerationConfigJson(error) => {
                write!(formatter, "cannot parse generation_config.json: {error}")
            }
            Self::MissingTextConfig => {
                write!(formatter, "config.json does not contain text_config")
            }
            Self::Tokenizer(error) => write!(formatter, "cannot load tokenizer: {error}"),
            Self::Prompt(error) => write!(formatter, "cannot render prompt: {error}"),
            Self::EmptyPrompt => write!(formatter, "prompt must contain at least one token"),
            Self::InvalidConfig(message) => write!(formatter, "invalid runtime config: {message}"),
            Self::WrongVectorLength {
                name,
                actual,
                expected,
            } => write!(
                formatter,
                "tensor {name:?} has {actual} values, expected {expected}"
            ),
            Self::VectorLengthMismatch { actual, expected } => {
                write!(formatter, "vector has {actual} values, expected {expected}")
            }
            Self::InvalidParameterBytes => {
                write!(formatter, "tensor parameter bytes are truncated")
            }
            Self::Image(message) => write!(formatter, "invalid image input: {message}"),
            Self::TokenOutOfRange(token) => {
                write!(formatter, "token id {token} exceeds the vocabulary")
            }
            Self::ContextLimit { requested, maximum } => {
                write!(
                    formatter,
                    "requested {requested} tokens but the context limit is {maximum}"
                )
            }
            Self::Unavailable(message) => write!(formatter, "native model unavailable: {message}"),
            Self::Streaming(message) => write!(formatter, "stream receiver unavailable: {message}"),
            Self::Preflight(error) => write!(formatter, "cannot inspect MTP capability: {error}"),
            Self::PrefixCachePoisoned => write!(formatter, "prefix cache lock is poisoned"),
            Self::MtpControllerPoisoned => write!(formatter, "MTP controller lock is poisoned"),
        }
    }
}

impl Error for NativeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Metal(error) => Some(error),
            Self::MissingTensor(_)
            | Self::NotQuantizedWeight(_)
            | Self::InvalidQuantizedWeight { .. }
            | Self::InvalidDenseWeight { .. }
            | Self::InvalidAffineTensor { .. }
            | Self::TensorByteLength { .. }
            | Self::InputDimension { .. }
            | Self::MissingMappedShard(_)
            | Self::DimensionOverflow(_)
            | Self::MissingTextConfig
            | Self::Tokenizer(_)
            | Self::Prompt(_)
            | Self::EmptyPrompt
            | Self::InvalidConfig(_)
            | Self::WrongVectorLength { .. }
            | Self::VectorLengthMismatch { .. }
            | Self::InvalidParameterBytes
            | Self::Image(_)
            | Self::TokenOutOfRange(_)
            | Self::ContextLimit { .. }
            | Self::Unavailable(_)
            | Self::Streaming(_)
            | Self::PrefixCachePoisoned
            | Self::MtpControllerPoisoned => None,
            Self::ConfigRead { source, .. } => Some(source),
            Self::ConfigJson(error) => Some(error),
            Self::GenerationConfigJson(error) => Some(error),
            Self::Preflight(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{PromptMessage, ThinkingConfig, ToolDefinition};

    fn tensor(dtype: &str, shape: &[u64], byte_len: u64) -> MlxTensor {
        MlxTensor {
            dtype: dtype.to_owned(),
            shape: shape.to_vec(),
            shard_index: 0,
            byte_offset: 0,
            byte_len,
        }
    }

    #[test]
    fn accepts_an_mlx_q4_affine_matrix_layout() {
        let matrix = Q4AffineMatrix::from_tensors(
            "layer.weight",
            tensor("U32", &[3, 16], 192),
            tensor("BF16", &[3, 2], 12),
            tensor("BF16", &[3, 2], 12),
        )
        .unwrap();

        assert_eq!(matrix.input_elements, 128);
        assert_eq!(matrix.output_rows, 3);
    }

    #[test]
    fn rejects_a_wrong_scale_layout() {
        let error = Q4AffineMatrix::from_tensors(
            "layer.weight",
            tensor("U32", &[3, 16], 192),
            tensor("BF16", &[3, 3], 18),
            tensor("BF16", &[3, 2], 12),
        )
        .unwrap_err();

        assert!(matches!(error, NativeError::InvalidAffineTensor { .. }));
    }

    #[test]
    fn mtp_eligibility_is_limited_to_greedy_text() {
        let request = GenerationRequest {
            messages: vec![PromptMessage::text(PromptRole::User, "hello")],
            tools: Vec::new(),
            tool_choice: ToolChoice::None,
            thinking: ThinkingConfig::DISABLED,
            max_tokens: 4,
            temperature: Some(0.0),
            top_p: Some(1.0),
            stop: Vec::new(),
        };
        assert!(mtp_request_is_eligible(&request, true));

        let mut with_tools = request.clone();
        with_tools.tools.push(ToolDefinition {
            name: "lookup".to_owned(),
            description: None,
            input_schema: serde_json::json!({"type": "object"}),
        });
        assert!(!mtp_request_is_eligible(&with_tools, true));

        let mut with_thinking = request.clone();
        with_thinking.thinking = ThinkingConfig::ENABLED;
        assert!(!mtp_request_is_eligible(&with_thinking, true));

        let mut sampled = request.clone();
        sampled.temperature = Some(0.2);
        assert!(!mtp_request_is_eligible(&sampled, true));

        let mut narrowed = request;
        narrowed.top_p = Some(0.9);
        assert!(!mtp_request_is_eligible(&narrowed, true));
    }

    #[test]
    fn mtp_defaults_to_a_single_draft_token() {
        assert_eq!(
            configured_mtp_draft_limit_with_override(2, None).unwrap(),
            1
        );
    }

    #[test]
    fn mtp_draft_override_can_restore_the_adapter_block_depth() {
        assert_eq!(
            configured_mtp_draft_limit_with_override(2, Some("2")).unwrap(),
            2
        );
    }

    #[test]
    fn splits_query_and_gate_for_each_attention_head() {
        let values = vec![
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0,
        ];

        let (query, gate) = split_query_and_gate(&values, 2, 3).unwrap();

        assert_eq!(query, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn selects_the_longest_matching_prefix() {
        let prefixes = [vec![1, 2], vec![1, 2, 3, 4], vec![9, 9]];
        let index = longest_prefix_index(
            prefixes
                .iter()
                .enumerate()
                .map(|(index, prefix)| (index, prefix.as_slice())),
            &[1, 2, 3, 4, 5],
        );
        assert_eq!(index, Some(1));
    }

    #[test]
    fn prefix_cache_admission_honors_limits() {
        let cache = PrefixCache::<()> {
            entries: VecDeque::new(),
            total_tokens: 0,
            max_entries: 2,
            max_tokens: 128,
            min_tokens: 8,
        };
        assert!(!cache.can_store(7));
        assert!(cache.can_store(8));
        assert!(cache.can_store(128));
        assert!(!cache.can_store(129));
    }

    #[test]
    fn prefix_cache_duplicate_replaces_without_double_counting() {
        let mut cache = PrefixCache::<()> {
            entries: VecDeque::new(),
            total_tokens: 0,
            max_entries: 3,
            max_tokens: 32,
            min_tokens: 1,
        };
        cache.insert(vec![1, 2, 3], vec![10.0], ());
        cache.insert(vec![4, 5], vec![20.0], ());
        cache.insert(vec![1, 2, 3], vec![30.0], ());

        assert_eq!(cache.total_tokens, 5);
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.entries.front().unwrap().token_ids, [1, 2, 3]);
        assert_eq!(cache.entries.front().unwrap().hidden, [30.0]);
    }

    #[test]
    fn prefix_cache_touch_makes_recent_entry_survive_lru_eviction() {
        let mut cache = PrefixCache::<()> {
            entries: VecDeque::new(),
            total_tokens: 0,
            max_entries: 2,
            max_tokens: 32,
            min_tokens: 1,
        };
        cache.insert(vec![1, 2], Vec::new(), ());
        cache.insert(vec![10, 11], Vec::new(), ());

        let index = cache.find_longest(&[1, 2, 99]).unwrap();
        cache.touch(index);
        cache.insert(vec![20, 21], Vec::new(), ());

        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.entries.front().unwrap().token_ids, [20, 21]);
        assert!(cache.entries.iter().any(|entry| entry.token_ids == [1, 2]));
        assert!(!cache
            .entries
            .iter()
            .any(|entry| entry.token_ids == [10, 11]));
        assert_eq!(cache.total_tokens, 4);
    }

    #[test]
    fn prefix_cache_evicts_until_total_token_budget_fits() {
        let mut cache = PrefixCache::<()> {
            entries: VecDeque::new(),
            total_tokens: 0,
            max_entries: 4,
            max_tokens: 5,
            min_tokens: 1,
        };
        cache.insert(vec![1, 2, 3], Vec::new(), ());
        cache.insert(vec![4, 5, 6], Vec::new(), ());

        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.entries.front().unwrap().token_ids, [4, 5, 6]);
        assert_eq!(cache.total_tokens, 3);
    }
}
