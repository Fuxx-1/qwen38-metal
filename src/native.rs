use crate::api::{
    EngineError, ExecutionKind, FinishReason, Generation, GenerationRequest, InferenceEngine,
    ModelDescriptor, PromptMessage, PromptRole,
};
use crate::metal_runtime::{MappedWeightBuffers, MetalRuntime, MetalRuntimeError};
use crate::model::{open_mlx_safetensors_dir, MlxTensor, MlxWeightStore};
use serde::Deserialize;
use std::error::Error;
use std::fmt;
use std::path::Path;
use tokenizers::Tokenizer;

const VALUES_PER_PACKED_WORD: u64 = 8;
const AFFINE_GROUP_SIZE: u64 = 64;
const DEFAULT_EOS_TOKEN_ID: u32 = 248_044;
const END_OF_MESSAGE_TOKEN_ID: u32 = 248_046;

#[derive(Debug, Deserialize)]
struct RuntimeConfig {
    #[serde(default)]
    text_config: Option<TextRuntimeConfig>,
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

    pub fn q4_affine_matvec(
        &self,
        runtime: &MetalRuntime,
        matrix: &Q4AffineMatrix,
        input: &[f32],
    ) -> Result<Vec<f32>, NativeError> {
        if input.len() as u64 != matrix.input_elements {
            return Err(NativeError::InputDimension {
                actual: input.len(),
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
        let result = if aligned {
            runtime.q4_affine_matvec_mapped(
                input,
                weight_buffer,
                matrix.weight.byte_offset,
                scale_buffer,
                matrix.scales.byte_offset,
                bias_buffer,
                matrix.biases.byte_offset,
                matrix.output_rows as usize,
            )
        } else {
            runtime.q4_affine_matvec_mapped_unaligned(
                input,
                weight_buffer,
                matrix.weight.byte_offset,
                scale_buffer,
                matrix.scales.byte_offset,
                bias_buffer,
                matrix.biases.byte_offset,
                matrix.output_rows as usize,
            )
        };
        result.map_err(NativeError::Metal)
    }

    pub fn mapped_shard_count(&self) -> usize {
        self.mapped.shard_count()
    }

    pub fn tensor_values_f32(&self, name: &str) -> Result<Vec<f32>, NativeError> {
        self.store
            .tensor_values_f32(name)
            .map_err(NativeError::Model)
    }
}

pub struct NativeEngine {
    descriptor: ModelDescriptor,
    runtime: MetalRuntime,
    weights: NativeWeights,
    tokenizer: Tokenizer,
    model: NativeModel,
    eos_token_ids: Vec<u32>,
}

impl NativeEngine {
    pub fn open(path: impl AsRef<Path>, model_id: impl Into<String>) -> Result<Self, NativeError> {
        let path = path.as_ref();
        let model_id = model_id.into();
        let config_bytes =
            std::fs::read(path.join("config.json")).map_err(|source| NativeError::ConfigRead {
                path: path.join("config.json"),
                source,
            })?;
        let config: RuntimeConfig =
            serde_json::from_slice(&config_bytes).map_err(NativeError::ConfigJson)?;
        let text_config = config.text_config.ok_or(NativeError::MissingTextConfig)?;
        validate_runtime_config(&text_config)?;

        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|error| NativeError::Tokenizer(error.to_string()))?;
        let runtime = MetalRuntime::new().map_err(NativeError::Metal)?;
        let weights = NativeWeights::open(path, &runtime)?;
        let model = NativeModel::load(&weights, text_config.clone())?;
        let context_tokens = u32::try_from(text_config.max_context())
            .map_err(|_| NativeError::DimensionOverflow("max_position_embeddings".to_owned()))?;
        let eos_token_ids = load_eos_token_ids(path, text_config.eos_token_id())?;

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
            eos_token_ids,
        })
    }

    pub fn mapped_shard_count(&self) -> usize {
        self.weights.mapped_shard_count()
    }

    fn render_prompt(&self, messages: &[PromptMessage]) -> Result<String, NativeError> {
        if messages.is_empty() {
            return Err(NativeError::EmptyPrompt);
        }
        let mut prompt = String::new();
        for message in messages {
            let role = match message.role {
                PromptRole::System => "system",
                PromptRole::User => "user",
                PromptRole::Assistant => "assistant",
            };
            prompt.push_str("<|im_start|>");
            prompt.push_str(role);
            prompt.push('\n');
            prompt.push_str(&message.content);
            prompt.push_str("<|im_end|>\n");
        }
        // The public API serves normal answers by closing the optional thinking block.
        prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        Ok(prompt)
    }

    fn tokenize(&self, messages: &[PromptMessage]) -> Result<Vec<u32>, NativeError> {
        let prompt = self.render_prompt(messages)?;
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

    fn generate_native(&self, request: GenerationRequest) -> Result<Generation, NativeError> {
        let prompt_ids = self.tokenize(&request.messages)?;
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

        let mut state = RuntimeState::new(&self.model);
        let mut hidden = Vec::new();
        for (position, token_id) in prompt_ids.iter().copied().enumerate() {
            hidden = self.model.forward_token(
                &self.runtime,
                &self.weights,
                &mut state,
                token_id,
                position,
            )?;
        }

        let mut generated_ids = Vec::new();
        let mut next_logits = self.model.logits(&self.runtime, &self.weights, &hidden)?;
        let mut finish_reason = FinishReason::Length;
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
            let position = prompt_ids.len() + step as usize;
            hidden = self.model.forward_token(
                &self.runtime,
                &self.weights,
                &mut state,
                token_id,
                position,
            )?;
            next_logits = self.model.logits(&self.runtime, &self.weights, &hidden)?;
            if let Some(stop_index) =
                token_sequence_stop_index(&self.tokenizer, &generated_ids, &request.stop)?
            {
                generated_ids.truncate(stop_index);
                finish_reason = FinishReason::StopSequence;
                break;
            }
            if step + 1 == request.max_tokens {
                finish_reason = FinishReason::Length;
            }
        }

        let text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|error| NativeError::Tokenizer(error.to_string()))?;
        let output_tokens = u32::try_from(generated_ids.len())
            .map_err(|_| NativeError::DimensionOverflow("output token count".to_owned()))?;
        Ok(Generation {
            text,
            input_tokens,
            output_tokens,
            finish_reason,
        })
    }
}

impl InferenceEngine for NativeEngine {
    fn descriptor(&self) -> ModelDescriptor {
        self.descriptor.clone()
    }

    fn estimate_prompt_tokens(&self, messages: &[PromptMessage]) -> Result<u32, EngineError> {
        self.tokenize(messages)
            .and_then(|ids| {
                u32::try_from(ids.len())
                    .map_err(|_| NativeError::DimensionOverflow("prompt token count".to_owned()))
            })
            .map_err(|error| EngineError::Failure(error.to_string()))
    }

    fn generate(&self, request: GenerationRequest) -> Result<Generation, EngineError> {
        self.generate_native(request).map_err(|error| match error {
            NativeError::ContextLimit { requested, maximum } => {
                EngineError::ContextLimit { requested, maximum }
            }
            NativeError::Unavailable(message) => EngineError::Unavailable(message),
            other => EngineError::Failure(other.to_string()),
        })
    }
}

struct NativeModel {
    config: TextRuntimeConfig,
    layers: Vec<LayerWeights>,
    embed_tokens: Q4AffineMatrix,
    lm_head: Q4AffineMatrix,
    model_norm: Vec<f32>,
}

#[allow(clippy::large_enum_variant)]
enum LayerWeights {
    Linear(LinearLayerWeights),
    Full(FullLayerWeights),
}

struct CommonLayerWeights {
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
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
    conv_weight: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
}

struct FullLayerWeights {
    common: CommonLayerWeights,
    q_proj: Q4AffineMatrix,
    k_proj: Q4AffineMatrix,
    v_proj: Q4AffineMatrix,
    o_proj: Q4AffineMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl NativeModel {
    fn load(weights: &NativeWeights, config: TextRuntimeConfig) -> Result<Self, NativeError> {
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

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("language_model.model.layers.{index}");
            let common = CommonLayerWeights {
                input_norm: load_vector(
                    weights,
                    &format!("{prefix}.input_layernorm.weight"),
                    config.hidden_size,
                )?,
                post_attention_norm: load_vector(
                    weights,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    config.hidden_size,
                )?,
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
                    conv_weight: load_vector(
                        weights,
                        &format!("{prefix}.linear_attn.conv1d.weight"),
                        (config.linear_num_key_heads * config.linear_key_head_dim * 2
                            + config.linear_num_value_heads * config.linear_value_head_dim)
                            * config.linear_conv_kernel_dim,
                    )?,
                    a_log: load_vector(
                        weights,
                        &format!("{prefix}.linear_attn.A_log"),
                        config.linear_num_value_heads,
                    )?,
                    dt_bias: load_vector(
                        weights,
                        &format!("{prefix}.linear_attn.dt_bias"),
                        config.linear_num_value_heads,
                    )?,
                    norm: load_vector(
                        weights,
                        &format!("{prefix}.linear_attn.norm.weight"),
                        config.linear_value_head_dim,
                    )?,
                };
                layers.push(LayerWeights::Linear(linear));
            } else {
                let full = FullLayerWeights {
                    common,
                    q_proj: weights.q4_matrix(&format!("{prefix}.self_attn.q_proj.weight"))?,
                    k_proj: weights.q4_matrix(&format!("{prefix}.self_attn.k_proj.weight"))?,
                    v_proj: weights.q4_matrix(&format!("{prefix}.self_attn.v_proj.weight"))?,
                    o_proj: weights.q4_matrix(&format!("{prefix}.self_attn.o_proj.weight"))?,
                    q_norm: load_vector(
                        weights,
                        &format!("{prefix}.self_attn.q_norm.weight"),
                        config.head_dim,
                    )?,
                    k_norm: load_vector(
                        weights,
                        &format!("{prefix}.self_attn.k_norm.weight"),
                        config.head_dim,
                    )?,
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

    fn forward_token(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        state: &mut RuntimeState,
        token_id: u32,
        position: usize,
    ) -> Result<Vec<f32>, NativeError> {
        if token_id as usize >= self.config.vocab_size {
            return Err(NativeError::TokenOutOfRange(token_id));
        }
        let mut hidden = dequantized_row(weights, &self.embed_tokens, token_id as usize)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let normalized = rms_norm(&hidden, layer.input_norm(), self.config.rms_norm_eps)?;
            let mixed = match layer {
                LayerWeights::Linear(linear) => linear.forward(
                    runtime,
                    weights,
                    &normalized,
                    &mut state.linear[layer_index],
                    self.config.rms_norm_eps,
                )?,
                LayerWeights::Full(full) => full.forward(
                    runtime,
                    weights,
                    &normalized,
                    &mut state.full[layer_index],
                    position,
                    self.config.rope(),
                    self.config.rms_norm_eps,
                )?,
            };
            add_in_place(&mut hidden, &mixed)?;
            let post_norm = rms_norm(
                &hidden,
                layer.post_attention_norm(),
                self.config.rms_norm_eps,
            )?;
            let gate = match layer {
                LayerWeights::Linear(linear) => {
                    weights.q4_affine_matvec(runtime, &linear.common.gate_proj, &post_norm)?
                }
                LayerWeights::Full(full) => {
                    weights.q4_affine_matvec(runtime, &full.common.gate_proj, &post_norm)?
                }
            };
            let up = match layer {
                LayerWeights::Linear(linear) => {
                    weights.q4_affine_matvec(runtime, &linear.common.up_proj, &post_norm)?
                }
                LayerWeights::Full(full) => {
                    weights.q4_affine_matvec(runtime, &full.common.up_proj, &post_norm)?
                }
            };
            let mut swiglu = Vec::with_capacity(gate.len());
            for (gate, up) in gate.into_iter().zip(up) {
                swiglu.push(silu(gate) * up);
            }
            let mlp = match layer {
                LayerWeights::Linear(linear) => {
                    weights.q4_affine_matvec(runtime, &linear.common.down_proj, &swiglu)?
                }
                LayerWeights::Full(full) => {
                    weights.q4_affine_matvec(runtime, &full.common.down_proj, &swiglu)?
                }
            };
            add_in_place(&mut hidden, &mlp)?;
        }
        Ok(hidden)
    }
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
    linear: Vec<LinearState>,
    full: Vec<FullState>,
}

struct LinearState {
    conv: Vec<f32>,
    recurrent: Vec<f32>,
}

struct FullState {
    keys: Vec<u16>,
    values: Vec<u16>,
}

impl RuntimeState {
    fn new(model: &NativeModel) -> Self {
        // Keep state indexed by original layer index. Non-applicable entries are empty.
        let mut linear_states = Vec::with_capacity(model.layers.len());
        let mut full_states = Vec::with_capacity(model.layers.len());
        for layer in model.layers.iter() {
            match layer {
                LayerWeights::Linear(layer) => linear_states.push(LinearState {
                    conv: vec![0.0; layer.configured_conv_width() * (layer.conv_kernel_size() - 1)],
                    recurrent: vec![0.0; layer.recurrent_len()],
                }),
                LayerWeights::Full(_) => linear_states.push(LinearState {
                    conv: Vec::new(),
                    recurrent: Vec::new(),
                }),
            }
            match layer {
                LayerWeights::Full(_) => full_states.push(FullState {
                    keys: Vec::new(),
                    values: Vec::new(),
                }),
                LayerWeights::Linear(_) => full_states.push(FullState {
                    keys: Vec::new(),
                    values: Vec::new(),
                }),
            }
        }
        Self {
            linear: linear_states,
            full: full_states,
        }
    }
}

impl LinearLayerWeights {
    fn configured_conv_width(&self) -> usize {
        self.in_proj_qkv.output_rows as usize
    }

    fn conv_kernel_size(&self) -> usize {
        self.conv_weight.len() / self.in_proj_qkv.output_rows as usize
    }

    fn recurrent_len(&self) -> usize {
        // q/k head dimension is derived from the qkv layout: qkv = 2*K + V.
        let value_dim = self.in_proj_z.output_rows as usize;
        let key_dim = (self.in_proj_qkv.output_rows as usize - value_dim) / 2;
        let key_heads = key_dim / 128;
        let value_heads = value_dim / 128;
        value_heads * 128 * (key_dim / key_heads)
    }

    fn forward(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        input: &[f32],
        state: &mut LinearState,
        eps: f32,
    ) -> Result<Vec<f32>, NativeError> {
        let qkv = weights.q4_affine_matvec(runtime, &self.in_proj_qkv, input)?;
        let z = weights.q4_affine_matvec(runtime, &self.in_proj_z, input)?;
        let b = weights.q4_affine_matvec(runtime, &self.in_proj_b, input)?;
        let a = weights.q4_affine_matvec(runtime, &self.in_proj_a, input)?;
        let mut conv_out = vec![0.0; qkv.len()];
        let kernel_size = self.conv_kernel_size();
        for channel in 0..qkv.len() {
            let state_offset = channel * (kernel_size - 1);
            let weight_offset = channel * kernel_size;
            let mut value = 0.0;
            for tap in 0..kernel_size - 1 {
                value += self.conv_weight[weight_offset + tap] * state.conv[state_offset + tap];
            }
            value += self.conv_weight[weight_offset + kernel_size - 1] * qkv[channel];
            conv_out[channel] = silu(value);
            if kernel_size > 1 {
                state.conv.copy_within(
                    state_offset + 1..state_offset + kernel_size - 1,
                    state_offset,
                );
                state.conv[state_offset + kernel_size - 2] = qkv[channel];
            }
        }
        let key_dim = (qkv.len() - z.len()) / 2;
        let value_dim = z.len();
        let key_heads = 16;
        let value_heads = value_dim / 128;
        let key_head_dim = key_dim / key_heads;
        let mut query = vec![0.0; key_dim];
        let mut key = vec![0.0; key_dim];
        let value = &conv_out[key_dim * 2..];
        query.copy_from_slice(&conv_out[..key_dim]);
        key.copy_from_slice(&conv_out[key_dim..key_dim * 2]);
        let inv_scale = (key_head_dim as f32).sqrt().recip();
        for head in 0..key_heads {
            let start = head * key_head_dim;
            let query_norm = rms_norm_slice(
                &query[start..start + key_head_dim],
                &vec![1.0; key_head_dim],
                eps,
            );
            let key_norm = rms_norm_slice(
                &key[start..start + key_head_dim],
                &vec![1.0; key_head_dim],
                eps,
            );
            query[start..start + key_head_dim]
                .iter_mut()
                .zip(query_norm)
                .for_each(|(destination, value)| *destination = value * inv_scale * inv_scale);
            key[start..start + key_head_dim]
                .iter_mut()
                .zip(key_norm)
                .for_each(|(destination, value)| *destination = value * inv_scale);
        }

        let beta: Vec<f32> = b.into_iter().map(sigmoid).collect();
        let mut decay = Vec::with_capacity(value_heads);
        for (head, a_value) in a.iter().take(value_heads).enumerate() {
            let gate = *a_value + self.dt_bias[head];
            decay.push((-self.a_log[head].exp() * softplus(gate)).exp());
        }
        let mut output = vec![0.0; value_dim];
        let repeat = value_heads / key_heads;
        for value_head in 0..value_heads {
            let key_head = value_head / repeat;
            let q_offset = key_head * key_head_dim;
            let v_offset = value_head * 128;
            for value_index in 0..128 {
                let state_offset = (value_head * 128 + value_index) * key_head_dim;
                let mut kv_mem = 0.0;
                for key_index in 0..key_head_dim {
                    state.recurrent[state_offset + key_index] *= decay[value_head];
                    kv_mem += state.recurrent[state_offset + key_index] * key[q_offset + key_index];
                }
                let delta = (value[v_offset + value_index] - kv_mem) * beta[value_head];
                let mut output_value = 0.0;
                for key_index in 0..key_head_dim {
                    state.recurrent[state_offset + key_index] += key[q_offset + key_index] * delta;
                    output_value +=
                        state.recurrent[state_offset + key_index] * query[q_offset + key_index];
                }
                output[v_offset + value_index] = output_value;
            }
        }
        for value_head in 0..value_heads {
            let offset = value_head * 128;
            let normalized = rms_norm_slice(&output[offset..offset + 128], &self.norm, eps);
            for index in 0..128 {
                output[offset + index] = normalized[index] * silu(z[offset + index]);
            }
        }
        let projected = weights.q4_affine_matvec(runtime, &self.out_proj, &output)?;
        Ok(projected)
    }
}

impl FullLayerWeights {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        runtime: &MetalRuntime,
        weights: &NativeWeights,
        input: &[f32],
        state: &mut FullState,
        position: usize,
        rope: RopeParameters,
        eps: f32,
    ) -> Result<Vec<f32>, NativeError> {
        let q_with_gate = weights.q4_affine_matvec(runtime, &self.q_proj, input)?;
        let (mut query, gate) =
            split_query_and_gate(&q_with_gate, self.num_attention_heads, self.head_dim)?;
        let mut key = weights.q4_affine_matvec(runtime, &self.k_proj, input)?;
        let value = weights.q4_affine_matvec(runtime, &self.v_proj, input)?;
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
            apply_rope(
                &mut query[offset..offset + head_dim],
                position,
                rotary_dim,
                rope.rope_theta,
            );
        }
        for head in 0..kv_heads {
            let offset = head * head_dim;
            let normalized = rms_norm_slice(&key[offset..offset + head_dim], &self.k_norm, eps);
            key[offset..offset + head_dim].copy_from_slice(&normalized);
            apply_rope(
                &mut key[offset..offset + head_dim],
                position,
                rotary_dim,
                rope.rope_theta,
            );
        }

        state.keys.extend(key.into_iter().map(f32_to_bf16));
        state.values.extend(value.into_iter().map(f32_to_bf16));
        let sequence_length = state.keys.len() / (kv_heads * head_dim);
        let mut attention_output = vec![0.0; num_heads * head_dim];
        let scale = (head_dim as f32).sqrt().recip();
        for head in 0..num_heads {
            let kv_head = head * kv_heads / num_heads;
            let q_offset = head * head_dim;
            let mut scores = Vec::with_capacity(sequence_length);
            for token in 0..sequence_length {
                let key_offset = (token * kv_heads + kv_head) * head_dim;
                let dot = (0..head_dim).fold(0.0, |sum, index| {
                    sum + query[q_offset + index] * bf16_to_f32(state.keys[key_offset + index])
                });
                scores.push(dot * scale);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denominator += *score;
            }
            let output_offset = head * head_dim;
            for (token, probability) in scores.iter().enumerate().take(sequence_length) {
                let probability = *probability / denominator.max(f32::MIN_POSITIVE);
                let value_offset = (token * kv_heads + kv_head) * head_dim;
                for index in 0..head_dim {
                    attention_output[output_offset + index] +=
                        probability * bf16_to_f32(state.values[value_offset + index]);
                }
            }
        }
        for head in 0..num_heads {
            let offset = head * head_dim;
            let gate_offset = head * head_dim;
            for index in 0..head_dim {
                attention_output[offset + index] *= sigmoid(gate[gate_offset + index]);
            }
        }
        weights.q4_affine_matvec(runtime, &self.o_proj, &attention_output)
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

fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, NativeError> {
    if input.len() != weight.len() {
        return Err(NativeError::VectorLengthMismatch {
            actual: weight.len(),
            expected: input.len(),
        });
    }
    Ok(rms_norm_slice(input, weight, eps))
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

fn apply_rope(values: &mut [f32], position: usize, rotary_dim: usize, theta: f32) {
    let half = rotary_dim / 2;
    for index in 0..half {
        let exponent = (2 * index) as f32 / rotary_dim as f32;
        let angle = position as f32 / theta.powf(exponent);
        let (sin, cos) = angle.sin_cos();
        let left = values[index];
        let right = values[index + half];
        values[index] = left * cos - right * sin;
        values[index + half] = right * cos + left * sin;
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
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
    TokenOutOfRange(u32),
    ContextLimit {
        requested: u32,
        maximum: u32,
    },
    Unavailable(String),
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
            | Self::InvalidAffineTensor { .. }
            | Self::TensorByteLength { .. }
            | Self::InputDimension { .. }
            | Self::MissingMappedShard(_)
            | Self::DimensionOverflow(_)
            | Self::MissingTextConfig
            | Self::Tokenizer(_)
            | Self::EmptyPrompt
            | Self::InvalidConfig(_)
            | Self::WrongVectorLength { .. }
            | Self::VectorLengthMismatch { .. }
            | Self::InvalidParameterBytes
            | Self::TokenOutOfRange(_)
            | Self::ContextLimit { .. }
            | Self::Unavailable(_) => None,
            Self::ConfigRead { source, .. } => Some(source),
            Self::ConfigJson(error) => Some(error),
            Self::GenerationConfigJson(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn splits_query_and_gate_for_each_attention_head() {
        let values = vec![
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0,
        ];

        let (query, gate) = split_query_and_gate(&values, 2, 3).unwrap();

        assert_eq!(query, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }
}
