use crate::metal::EMBEDDED_LIBRARY;
use crate::model::MlxWeightStore;
use crate::mps::{self, MpsFp16Gemm, MpsMatrix};
use metal::{ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use std::error::Error;
use std::fmt;
use std::mem::size_of;
use std::sync::Mutex;

const QUANT_BITS: usize = 4;
const VALUES_PER_PACKED_WORD: usize = 32 / QUANT_BITS;
const AFFINE_GROUP_SIZE: usize = 64;
const THREADS_PER_THREADGROUP: u64 = 256;
const Q4_PREFILL_BATCH_TILE: usize = 8;
const Q4_PREFILL_OUTPUT_TILE: usize = 32;
const Q4_PREFILL_INPUT_TILE_FLOATS: usize = Q4_PREFILL_BATCH_TILE * AFFINE_GROUP_SIZE;
const Q4_PREFILL_PACKED_TILE_WORDS: usize =
    Q4_PREFILL_OUTPUT_TILE * (AFFINE_GROUP_SIZE / VALUES_PER_PACKED_WORD);
const Q4_PREFILL_AFFINE_TILE_FLOATS: usize = Q4_PREFILL_OUTPUT_TILE * 2;
const Q4_SIMDGROUP_PREFILL_MIN_BATCH: usize = 64;
const Q4_SIMDGROUP_PREFILL_BATCH_TILE: usize = 64;
const Q4_SIMDGROUP_PREFILL_OUTPUT_TILE: usize = 8;
const Q4_SIMDGROUP_PREFILL_K_TILE: usize = 64;
const Q4_SIMDGROUP_PREFILL_INPUT_HALF: usize =
    Q4_SIMDGROUP_PREFILL_BATCH_TILE * Q4_SIMDGROUP_PREFILL_K_TILE;
const Q4_SIMDGROUP_PREFILL_WEIGHT_HALF: usize =
    Q4_SIMDGROUP_PREFILL_K_TILE * Q4_SIMDGROUP_PREFILL_OUTPUT_TILE;
const Q4_SIMDGROUP_PREFILL_OUTPUT_FLOATS: usize =
    Q4_SIMDGROUP_PREFILL_BATCH_TILE * Q4_SIMDGROUP_PREFILL_OUTPUT_TILE;
// The 1024-thread tile amortizes activation traffic only once prompts reach
// a few thousand rows; below that point its lower occupancy is slower.
const Q4_SIMDGROUP_WIDE_PREFILL_MIN_BATCH: usize = 3072;
const Q4_SIMDGROUP_WIDE_PREFILL_BATCH_TILE: usize = 128;
const Q4_SIMDGROUP_WIDE_PREFILL_OUTPUT_TILE: usize = 16;
const Q4_SIMDGROUP_WIDE_PREFILL_K_TILE: usize = 64;
const Q4_SIMDGROUP_WIDE_PREFILL_INPUT_HALF: usize =
    Q4_SIMDGROUP_WIDE_PREFILL_BATCH_TILE * Q4_SIMDGROUP_WIDE_PREFILL_K_TILE;
const Q4_SIMDGROUP_WIDE_PREFILL_WEIGHT_HALF: usize =
    Q4_SIMDGROUP_WIDE_PREFILL_K_TILE * Q4_SIMDGROUP_WIDE_PREFILL_OUTPUT_TILE;
const Q4_SIMDGROUP_WIDE_PREFILL_OUTPUT_FLOATS: usize =
    Q4_SIMDGROUP_WIDE_PREFILL_BATCH_TILE * Q4_SIMDGROUP_WIDE_PREFILL_OUTPUT_TILE;
const Q4_SIMDGROUP_WIDE_PREFILL_THREADS: u64 = 1024;
// MPS uses Apple's precompiled matrix kernels. At this size, temporary FP16
// expansion is much cheaper than executing a Q4 tile kernel for every panel.
const Q4_MPS_PREFILL_MIN_BATCH: usize = 256;
const Q4_MPS_THREADS_X: u64 = 32;
const Q4_MPS_THREADS_Y: u64 = 8;
const Q4_DECODE_OUTPUT_TILE: usize = 8;
const Q4_DECODE_SHARED_MAX_INPUT_ELEMENTS: usize = 6_144;
const Q4_DECODE_TILED_INPUT_ELEMENTS: usize = 2_048;

pub struct MetalRuntime {
    device: Device,
    command_queue: metal::CommandQueue,
    q4_affine_matmul: ComputePipelineState,
    q4_affine_matmul_unaligned: ComputePipelineState,
    q4_affine_matvec_simd: ComputePipelineState,
    q4_affine_matvec_simd_unaligned: ComputePipelineState,
    q4_affine_matvec_mlx_fast: ComputePipelineState,
    q4_affine_matvec_mlx_fast_unaligned: ComputePipelineState,
    q4_affine_matvec_shared: ComputePipelineState,
    q4_affine_matvec_shared_unaligned: ComputePipelineState,
    q4_affine_matvec_tiled: ComputePipelineState,
    q4_affine_matvec_tiled_unaligned: ComputePipelineState,
    q4_affine_matmul_simdgroup: ComputePipelineState,
    q4_affine_matmul_simdgroup_wide: ComputePipelineState,
    q4_affine_dequantize_f16: ComputePipelineState,
    q4_affine_dequantize_f16_unaligned: ComputePipelineState,
    f32_to_f16: ComputePipelineState,
    f16_to_f32: ComputePipelineState,
    swiglu_rows: ComputePipelineState,
    swiglu_half_rows: ComputePipelineState,
    swiglu_half_split_rows: ComputePipelineState,
    rms_norm: ComputePipelineState,
    add_in_place: ComputePipelineState,
    bf16_gemm: ComputePipelineState,
    vision_attention_scores: ComputePipelineState,
    vision_attention_values: ComputePipelineState,
    deltanet_conv: ComputePipelineState,
    deltanet_prepare: ComputePipelineState,
    deltanet_recurrence: ComputePipelineState,
    deltanet_gate_norm: ComputePipelineState,
    deltanet_prefill: ComputePipelineState,
    q8_kv_append: ComputePipelineState,
    q8_kv_append_prefill: ComputePipelineState,
    gqa_prepare_query: ComputePipelineState,
    gqa_prepare_key: ComputePipelineState,
    gqa_q8_scores: ComputePipelineState,
    gqa_q8_values: ComputePipelineState,
    gqa_q8_prefill_attention: ComputePipelineState,
    q4_activations: Mutex<Q4ActivationPool>,
    language_activations: Mutex<LanguageActivationPool>,
    fast_q4_prefill: bool,
    fast_q4_decode: bool,
    mlx_q4_decode: bool,
    mps_q4_prefill: bool,
    mps_q4_mlp_fusion: bool,
}

/// A borrowed description of one mapped Q4 affine projection. A batch shares
/// its input activation and commits all projections in one command buffer.
#[derive(Clone, Copy)]
pub struct MappedQ4AffineJob<'a> {
    weights: &'a metal::Buffer,
    weight_offset: u64,
    scales: &'a metal::Buffer,
    scale_offset: u64,
    biases: &'a metal::Buffer,
    bias_offset: u64,
    output_rows: usize,
    aligned: bool,
}

impl<'a> MappedQ4AffineJob<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weights: &'a metal::Buffer,
        weight_offset: u64,
        scales: &'a metal::Buffer,
        scale_offset: u64,
        biases: &'a metal::Buffer,
        bias_offset: u64,
        output_rows: usize,
        aligned: bool,
    ) -> Self {
        Self {
            weights,
            weight_offset,
            scales,
            scale_offset,
            biases,
            bias_offset,
            output_rows,
            aligned,
        }
    }
}

struct ReusableBuffer {
    buffer: metal::Buffer,
    capacity_bytes: u64,
}

#[derive(Default)]
struct Q4ActivationPool {
    input: Option<ReusableBuffer>,
    outputs: Vec<Option<ReusableBuffer>>,
    input_half: Option<ReusableBuffer>,
    weights_half: Option<ReusableBuffer>,
    half_slots: Vec<Option<ReusableBuffer>>,
}

#[derive(Default)]
struct LanguageActivationPool {
    slots: Vec<Option<ReusableBuffer>>,
    scores: Option<ReusableBuffer>,
}

/// An immutable FP32 vector used by the GPU-resident decode path. Layer norm
/// weights are uploaded once while the model is opened instead of once per
/// generated token.
pub struct MetalF32Buffer {
    buffer: metal::Buffer,
    elements: usize,
}

/// Per-request decode activations. These buffers deliberately do not use the
/// shared runtime pools: concurrent requests need isolated residual streams,
/// while all immutable Q4 weights remain file-mapped and shared.
pub struct MetalDecodeState {
    hidden_elements: usize,
    hidden: ReusableBuffer,
    normalized: ReusableBuffer,
    post_norm: ReusableBuffer,
    mixed: ReusableBuffer,
    qkv: Option<ReusableBuffer>,
    z: Option<ReusableBuffer>,
    b: Option<ReusableBuffer>,
    a: Option<ReusableBuffer>,
    convolved: Option<ReusableBuffer>,
    delta_output: Option<ReusableBuffer>,
    gate: Option<ReusableBuffer>,
    up: Option<ReusableBuffer>,
    swiglu: Option<ReusableBuffer>,
    scores: Option<ReusableBuffer>,
}

/// Borrowed immutable inputs for one GPU-resident DeltaNet transformer layer.
/// The activation buffers live in `MetalDecodeState`; this descriptor only
/// points at mapped Q4 weights and request-local recurrent state.
#[derive(Clone, Copy)]
pub struct MetalDecodeLinearLayer<'a> {
    input_norm: &'a MetalF32Buffer,
    post_attention_norm: &'a MetalF32Buffer,
    qkv: MappedQ4AffineJob<'a>,
    z: MappedQ4AffineJob<'a>,
    b: MappedQ4AffineJob<'a>,
    a: MappedQ4AffineJob<'a>,
    out_proj: MappedQ4AffineJob<'a>,
    delta_weights: &'a MetalDeltaNetWeights,
    delta_state: &'a MetalDeltaNetState,
    gate_proj: MappedQ4AffineJob<'a>,
    up_proj: MappedQ4AffineJob<'a>,
    down_proj: MappedQ4AffineJob<'a>,
}

/// Immutable geometry and positional data for one full-attention decode step.
/// M-RoPE axes are passed explicitly so text and multimodal prompt positions
/// share the same GPU path.
#[derive(Debug, Clone, Copy)]
pub struct MetalGqaDecodeConfig {
    pub num_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub position: [u32; 3],
    pub section1: u32,
    pub section2: u32,
    pub has_mrope_sections: bool,
    pub rope_theta: f32,
}

#[derive(Clone, Copy)]
struct MetalGqaDecodeConfigU32 {
    num_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    position0: u32,
    position1: u32,
    position2: u32,
    section1: u32,
    section2: u32,
    has_mrope_sections: u32,
    rope_theta: f32,
}

impl MetalGqaDecodeConfig {
    fn as_u32(self) -> Result<MetalGqaDecodeConfigU32, MetalRuntimeError> {
        validate_attention_shape(self.kv_heads, self.head_dim)?;
        if self.num_heads == 0
            || self.num_heads % self.kv_heads != 0
            || self.rotary_dim > self.head_dim
            || self.rotary_dim % 2 != 0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(MetalRuntimeError::InvalidAttentionShape);
        }
        Ok(MetalGqaDecodeConfigU32 {
            num_heads: u32::try_from(self.num_heads)
                .map_err(|_| MetalRuntimeError::InvalidAttentionShape)?,
            kv_heads: u32::try_from(self.kv_heads)
                .map_err(|_| MetalRuntimeError::InvalidAttentionShape)?,
            head_dim: u32::try_from(self.head_dim)
                .map_err(|_| MetalRuntimeError::InvalidAttentionShape)?,
            rotary_dim: u32::try_from(self.rotary_dim)
                .map_err(|_| MetalRuntimeError::InvalidAttentionShape)?,
            position0: self.position[0],
            position1: self.position[1],
            position2: self.position[2],
            section1: self.section1,
            section2: self.section2,
            has_mrope_sections: u32::from(self.has_mrope_sections),
            rope_theta: self.rope_theta,
        })
    }
}

/// Borrowed weights for one GPU-resident full-attention transformer layer.
/// Its Q8 KV state remains request-local and is passed to the execution call.
#[derive(Clone, Copy)]
pub struct MetalDecodeFullLayer<'a> {
    input_norm: &'a MetalF32Buffer,
    post_attention_norm: &'a MetalF32Buffer,
    q_proj: MappedQ4AffineJob<'a>,
    k_proj: MappedQ4AffineJob<'a>,
    v_proj: MappedQ4AffineJob<'a>,
    o_proj: MappedQ4AffineJob<'a>,
    q_norm: &'a MetalF32Buffer,
    k_norm: &'a MetalF32Buffer,
    gqa: MetalGqaDecodeConfig,
    gate_proj: MappedQ4AffineJob<'a>,
    up_proj: MappedQ4AffineJob<'a>,
    down_proj: MappedQ4AffineJob<'a>,
}

impl<'a> MetalDecodeFullLayer<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_norm: &'a MetalF32Buffer,
        post_attention_norm: &'a MetalF32Buffer,
        q_proj: MappedQ4AffineJob<'a>,
        k_proj: MappedQ4AffineJob<'a>,
        v_proj: MappedQ4AffineJob<'a>,
        o_proj: MappedQ4AffineJob<'a>,
        q_norm: &'a MetalF32Buffer,
        k_norm: &'a MetalF32Buffer,
        gqa: MetalGqaDecodeConfig,
        gate_proj: MappedQ4AffineJob<'a>,
        up_proj: MappedQ4AffineJob<'a>,
        down_proj: MappedQ4AffineJob<'a>,
    ) -> Self {
        Self {
            input_norm,
            post_attention_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            gqa,
            gate_proj,
            up_proj,
            down_proj,
        }
    }
}

/// One transformer layer in a GPU-resident decode graph. Full-attention
/// layers retain their own request-local KV cache, while every layer shares
/// the decode activation stream.
pub enum MetalDecodeLayer<'a> {
    Linear(MetalDecodeLinearLayer<'a>),
    Full(MetalDecodeFullLayer<'a>, &'a mut Q8KvState),
}

impl<'a> MetalDecodeLinearLayer<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_norm: &'a MetalF32Buffer,
        post_attention_norm: &'a MetalF32Buffer,
        qkv: MappedQ4AffineJob<'a>,
        z: MappedQ4AffineJob<'a>,
        b: MappedQ4AffineJob<'a>,
        a: MappedQ4AffineJob<'a>,
        out_proj: MappedQ4AffineJob<'a>,
        delta_weights: &'a MetalDeltaNetWeights,
        delta_state: &'a MetalDeltaNetState,
        gate_proj: MappedQ4AffineJob<'a>,
        up_proj: MappedQ4AffineJob<'a>,
        down_proj: MappedQ4AffineJob<'a>,
    ) -> Self {
        Self {
            input_norm,
            post_attention_norm,
            qkv,
            z,
            b,
            a,
            out_proj,
            delta_weights,
            delta_state,
            gate_proj,
            up_proj,
            down_proj,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeltaNetConfig {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
}

impl DeltaNetConfig {
    fn channels(self) -> Result<usize, MetalRuntimeError> {
        self.key_heads
            .checked_mul(self.key_head_dim)
            .and_then(|keys| keys.checked_mul(2))
            .and_then(|keys| {
                self.value_heads
                    .checked_mul(self.value_head_dim)
                    .and_then(|values| keys.checked_add(values))
            })
            .ok_or(MetalRuntimeError::DimensionOverflow(
                "DeltaNet channel count",
            ))
    }

    fn recurrent_elements(self) -> Result<usize, MetalRuntimeError> {
        self.value_heads
            .checked_mul(self.value_head_dim)
            .and_then(|values| values.checked_mul(self.key_head_dim))
            .ok_or(MetalRuntimeError::DimensionOverflow(
                "DeltaNet recurrent state elements",
            ))
    }

    fn as_u32(self) -> Result<DeltaNetConfigU32, MetalRuntimeError> {
        Ok(DeltaNetConfigU32 {
            key_heads: u32::try_from(self.key_heads)
                .map_err(|_| MetalRuntimeError::DimensionOverflow("DeltaNet key heads"))?,
            value_heads: u32::try_from(self.value_heads)
                .map_err(|_| MetalRuntimeError::DimensionOverflow("DeltaNet value heads"))?,
            key_head_dim: u32::try_from(self.key_head_dim)
                .map_err(|_| MetalRuntimeError::DimensionOverflow("DeltaNet key head dimension"))?,
            value_head_dim: u32::try_from(self.value_head_dim).map_err(|_| {
                MetalRuntimeError::DimensionOverflow("DeltaNet value head dimension")
            })?,
            conv_kernel_size: u32::try_from(self.conv_kernel_size)
                .map_err(|_| MetalRuntimeError::DimensionOverflow("DeltaNet convolution kernel"))?,
        })
    }
}

#[derive(Clone, Copy)]
struct DeltaNetConfigU32 {
    key_heads: u32,
    value_heads: u32,
    key_head_dim: u32,
    value_head_dim: u32,
    conv_kernel_size: u32,
}

pub struct MetalDeltaNetWeights {
    config: DeltaNetConfig,
    conv_weight: metal::Buffer,
    a_log: metal::Buffer,
    dt_bias: metal::Buffer,
    norm: metal::Buffer,
}

pub struct MetalDeltaNetState {
    conv: metal::Buffer,
    recurrent: metal::Buffer,
}

pub struct Q8KvState {
    keys: metal::Buffer,
    values: metal::Buffer,
    key_scales: metal::Buffer,
    value_scales: metal::Buffer,
    capacity_tokens: usize,
    sequence_length: usize,
    kv_heads: usize,
    head_dim: usize,
}

fn pipeline_for(
    device: &Device,
    library: &metal::Library,
    name: &str,
) -> Result<ComputePipelineState, MetalRuntimeError> {
    let function = library
        .get_function(name, None)
        .map_err(MetalRuntimeError::Function)?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalRuntimeError::Pipeline)
}

/// GPU views of read-only safetensors mappings. `MlxWeightStore` must outlive
/// this value because Metal receives the mappings by pointer without copying.
pub struct MappedWeightBuffers {
    buffers: Vec<MappedWeightBuffer>,
}

struct MappedWeightBuffer {
    buffer: metal::Buffer,
    base_address: usize,
}

impl MappedWeightBuffers {
    pub fn buffer(&self, shard_index: usize) -> Option<&metal::Buffer> {
        self.buffers.get(shard_index).map(|entry| &entry.buffer)
    }

    pub fn shard_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn offset_is_aligned(
        &self,
        shard_index: usize,
        byte_offset: u64,
        alignment: usize,
    ) -> Option<bool> {
        let entry = self.buffers.get(shard_index)?;
        let offset = usize::try_from(byte_offset).ok()?;
        let address = entry.base_address.checked_add(offset)?;
        Some(address % alignment == 0)
    }
}

impl MetalRuntime {
    pub fn new() -> Result<Self, MetalRuntimeError> {
        let device = Device::system_default().ok_or(MetalRuntimeError::NoDevice)?;
        let library = device
            .new_library_with_data(EMBEDDED_LIBRARY)
            .map_err(MetalRuntimeError::Library)?;
        let q4_affine_matmul = pipeline_for(&device, &library, "qwen38_q4_affine_matmul")?;
        let q4_affine_matmul_unaligned =
            pipeline_for(&device, &library, "qwen38_q4_affine_matmul_unaligned")?;
        let q4_affine_matvec_simd =
            pipeline_for(&device, &library, "qwen38_q4_affine_matvec_simd")?;
        let q4_affine_matvec_simd_unaligned =
            pipeline_for(&device, &library, "qwen38_q4_affine_matvec_simd_unaligned")?;
        let q4_affine_matvec_mlx_fast =
            pipeline_for(&device, &library, "qwen38_q4_affine_matvec_mlx_fast")?;
        let q4_affine_matvec_mlx_fast_unaligned = pipeline_for(
            &device,
            &library,
            "qwen38_q4_affine_matvec_mlx_fast_unaligned",
        )?;
        let q4_affine_matvec_shared =
            pipeline_for(&device, &library, "qwen38_q4_affine_matvec_shared")?;
        let q4_affine_matvec_shared_unaligned = pipeline_for(
            &device,
            &library,
            "qwen38_q4_affine_matvec_shared_unaligned",
        )?;
        let q4_affine_matvec_tiled =
            pipeline_for(&device, &library, "qwen38_q4_affine_matvec_tiled")?;
        let q4_affine_matvec_tiled_unaligned =
            pipeline_for(&device, &library, "qwen38_q4_affine_matvec_tiled_unaligned")?;
        let q4_affine_matmul_simdgroup =
            pipeline_for(&device, &library, "qwen38_q4_affine_matmul_simdgroup")?;
        let q4_affine_matmul_simdgroup_wide =
            pipeline_for(&device, &library, "qwen38_q4_affine_matmul_simdgroup_wide")?;
        let q4_affine_dequantize_f16 =
            pipeline_for(&device, &library, "qwen38_q4_affine_dequantize_f16")?;
        let q4_affine_dequantize_f16_unaligned = pipeline_for(
            &device,
            &library,
            "qwen38_q4_affine_dequantize_f16_unaligned",
        )?;
        let f32_to_f16 = pipeline_for(&device, &library, "qwen38_f32_to_f16")?;
        let f16_to_f32 = pipeline_for(&device, &library, "qwen38_f16_to_f32")?;
        let swiglu_rows = pipeline_for(&device, &library, "qwen38_swiglu_rows")?;
        let swiglu_half_rows = pipeline_for(&device, &library, "qwen38_swiglu_half_rows")?;
        let swiglu_half_split_rows =
            pipeline_for(&device, &library, "qwen38_swiglu_half_split_rows")?;
        let rms_norm = pipeline_for(&device, &library, "qwen38_rms_norm")?;
        let add_in_place = pipeline_for(&device, &library, "qwen38_add_in_place")?;
        let bf16_gemm_function = library
            .get_function("qwen38_bf16_gemm", None)
            .map_err(MetalRuntimeError::Function)?;
        let bf16_gemm = device
            .new_compute_pipeline_state_with_function(&bf16_gemm_function)
            .map_err(MetalRuntimeError::Pipeline)?;
        let vision_attention_scores_function = library
            .get_function("qwen38_vision_attention_scores", None)
            .map_err(MetalRuntimeError::Function)?;
        let vision_attention_scores = device
            .new_compute_pipeline_state_with_function(&vision_attention_scores_function)
            .map_err(MetalRuntimeError::Pipeline)?;
        let vision_attention_values_function = library
            .get_function("qwen38_vision_attention_values", None)
            .map_err(MetalRuntimeError::Function)?;
        let vision_attention_values = device
            .new_compute_pipeline_state_with_function(&vision_attention_values_function)
            .map_err(MetalRuntimeError::Pipeline)?;
        let deltanet_conv = pipeline_for(&device, &library, "qwen38_deltanet_conv")?;
        let deltanet_prepare = pipeline_for(&device, &library, "qwen38_deltanet_prepare")?;
        let deltanet_recurrence = pipeline_for(&device, &library, "qwen38_deltanet_recurrence")?;
        let deltanet_gate_norm = pipeline_for(&device, &library, "qwen38_deltanet_gate_norm")?;
        let deltanet_prefill = pipeline_for(&device, &library, "qwen38_deltanet_prefill")?;
        let q8_kv_append = pipeline_for(&device, &library, "qwen38_q8_kv_append")?;
        let q8_kv_append_prefill = pipeline_for(&device, &library, "qwen38_q8_kv_append_prefill")?;
        let gqa_prepare_query = pipeline_for(&device, &library, "qwen38_gqa_prepare_query")?;
        let gqa_prepare_key = pipeline_for(&device, &library, "qwen38_gqa_prepare_key")?;
        let gqa_q8_scores = pipeline_for(&device, &library, "qwen38_gqa_q8_scores")?;
        let gqa_q8_values = pipeline_for(&device, &library, "qwen38_gqa_q8_values")?;
        let gqa_q8_prefill_attention =
            pipeline_for(&device, &library, "qwen38_gqa_q8_prefill_attention")?;

        let required_threadgroup_pipelines = [
            &q4_affine_matmul,
            &q4_affine_matmul_unaligned,
            &q4_affine_matvec_shared,
            &q4_affine_matvec_shared_unaligned,
            &q4_affine_matvec_tiled,
            &q4_affine_matvec_tiled_unaligned,
        ];
        let available_threads = required_threadgroup_pipelines
            .iter()
            .map(|pipeline| pipeline.max_total_threads_per_threadgroup())
            .min()
            .expect("at least one threadgroup pipeline is required");
        if available_threads < THREADS_PER_THREADGROUP {
            return Err(MetalRuntimeError::UnsupportedThreadgroupLimit {
                available: available_threads,
                required: THREADS_PER_THREADGROUP,
            });
        }

        let mps_q4_prefill =
            std::env::var_os("QWEN38_DISABLE_MPS_PREFILL").is_none() && mps::is_available(&device);
        Ok(Self {
            command_queue: device.new_command_queue(),
            device,
            q4_affine_matmul,
            q4_affine_matmul_unaligned,
            q4_affine_matvec_simd,
            q4_affine_matvec_simd_unaligned,
            q4_affine_matvec_mlx_fast,
            q4_affine_matvec_mlx_fast_unaligned,
            q4_affine_matvec_shared,
            q4_affine_matvec_shared_unaligned,
            q4_affine_matvec_tiled,
            q4_affine_matvec_tiled_unaligned,
            q4_affine_matmul_simdgroup,
            q4_affine_matmul_simdgroup_wide,
            q4_affine_dequantize_f16,
            q4_affine_dequantize_f16_unaligned,
            f32_to_f16,
            f16_to_f32,
            swiglu_rows,
            swiglu_half_rows,
            swiglu_half_split_rows,
            rms_norm,
            add_in_place,
            bf16_gemm,
            vision_attention_scores,
            vision_attention_values,
            deltanet_conv,
            deltanet_prepare,
            deltanet_recurrence,
            deltanet_gate_norm,
            deltanet_prefill,
            q8_kv_append,
            q8_kv_append_prefill,
            gqa_prepare_query,
            gqa_prepare_key,
            gqa_q8_scores,
            gqa_q8_values,
            gqa_q8_prefill_attention,
            q4_activations: Mutex::new(Q4ActivationPool::default()),
            language_activations: Mutex::new(LanguageActivationPool::default()),
            fast_q4_prefill: std::env::var_os("QWEN38_DISABLE_FAST_PREFILL").is_none(),
            // The shared-activation prototype is numerically valid but loses
            // occupancy on M4 Pro. Keep it available for targeted hardware
            // experiments without making the proven SIMD path regress.
            fast_q4_decode: std::env::var_os("QWEN38_ENABLE_SHARED_DECODE").is_some(),
            // Kept behind an opt-in while benchmarked against the established
            // one-row SIMD path on the target M4 Pro.
            mlx_q4_decode: std::env::var_os("QWEN38_ENABLE_MLX_DECODE").is_some(),
            mps_q4_prefill,
            mps_q4_mlp_fusion: std::env::var_os("QWEN38_ENABLE_MPS_MLP_FUSION").is_some(),
        })
    }

    pub fn q4_affine_matvec(
        &self,
        input: &[f32],
        packed_weights: &[u32],
        scales: &[u16],
        biases: &[u16],
        output_rows: usize,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        let shape = MatvecShape::validate(input, packed_weights, scales, biases, output_rows)?;
        let input_buffer = self.buffer_from_slice(input)?;
        let weights_buffer = self.buffer_from_slice(packed_weights)?;
        let scales_buffer = self.buffer_from_slice(scales)?;
        let biases_buffer = self.buffer_from_slice(biases)?;
        self.dispatch_q4_affine_matvec(
            &input_buffer,
            &weights_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
            output_rows,
            shape.words_per_row,
        )
    }

    pub fn map_weight_store(
        &self,
        store: &MlxWeightStore,
    ) -> Result<MappedWeightBuffers, MetalRuntimeError> {
        let mut buffers = Vec::with_capacity(store.shard_count());
        for shard_index in 0..store.shard_count() {
            let bytes = store
                .shard_data(shard_index)
                .ok_or(MetalRuntimeError::MissingMappedShard(shard_index))?;
            if bytes.is_empty() {
                return Err(MetalRuntimeError::EmptyBuffer);
            }
            let byte_len = u64::try_from(bytes.len())
                .map_err(|_| MetalRuntimeError::DimensionOverflow("mapped shard byte length"))?;
            let buffer = self.device.new_buffer_with_bytes_no_copy(
                bytes.as_ptr().cast(),
                byte_len,
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeDefaultCache,
                None,
            );
            buffers.push(MappedWeightBuffer {
                buffer,
                base_address: bytes.as_ptr() as usize,
            });
        }
        Ok(MappedWeightBuffers { buffers })
    }

    /// Uploads a small immutable vector once for reuse by the decode graph.
    /// This is primarily used for per-layer RMSNorm scales.
    pub fn create_f32_buffer(&self, values: &[f32]) -> Result<MetalF32Buffer, MetalRuntimeError> {
        Ok(MetalF32Buffer {
            buffer: self.buffer_from_slice(values)?,
            elements: values.len(),
        })
    }

    /// Allocates a request-local residual stream. Scratch buffers grow lazily
    /// from the exact projection sizes of the first linear-attention layer.
    pub fn create_decode_state(
        &self,
        hidden_elements: usize,
    ) -> Result<MetalDecodeState, MetalRuntimeError> {
        let hidden_bytes = checked_byte_len::<f32>(hidden_elements)?;
        Ok(MetalDecodeState {
            hidden_elements,
            hidden: shared_reusable_buffer(&self.device, hidden_bytes)?,
            normalized: shared_reusable_buffer(&self.device, hidden_bytes)?,
            post_norm: shared_reusable_buffer(&self.device, hidden_bytes)?,
            mixed: shared_reusable_buffer(&self.device, hidden_bytes)?,
            qkv: None,
            z: None,
            b: None,
            a: None,
            convolved: None,
            delta_output: None,
            gate: None,
            up: None,
            swiglu: None,
            scores: None,
        })
    }

    pub fn write_decode_hidden(
        &self,
        state: &mut MetalDecodeState,
        values: &[f32],
    ) -> Result<(), MetalRuntimeError> {
        if values.len() != state.hidden_elements {
            return Err(MetalRuntimeError::WrongLength {
                name: "decode hidden activation",
                actual: values.len(),
                expected: state.hidden_elements,
            });
        }
        copy_slice_to_buffer(&state.hidden.buffer, values);
        Ok(())
    }

    pub fn read_decode_hidden(
        &self,
        state: &MetalDecodeState,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        if state.hidden_elements == 0 {
            return Err(MetalRuntimeError::EmptyDimension);
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                state.hidden.buffer.contents().cast::<f32>(),
                state.hidden_elements,
            )
            .to_vec()
        })
    }

    /// Executes one complete linear-attention transformer layer. This is the
    /// single-layer convenience form of `decode_linear_layers`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_linear_layer(
        &self,
        state: &mut MetalDecodeState,
        input_norm: &MetalF32Buffer,
        post_attention_norm: &MetalF32Buffer,
        qkv: &MappedQ4AffineJob<'_>,
        z: &MappedQ4AffineJob<'_>,
        b: &MappedQ4AffineJob<'_>,
        a: &MappedQ4AffineJob<'_>,
        out_proj: &MappedQ4AffineJob<'_>,
        delta_weights: &MetalDeltaNetWeights,
        delta_state: &MetalDeltaNetState,
        gate_proj: &MappedQ4AffineJob<'_>,
        up_proj: &MappedQ4AffineJob<'_>,
        down_proj: &MappedQ4AffineJob<'_>,
        epsilon: f32,
    ) -> Result<(), MetalRuntimeError> {
        let layer = MetalDecodeLinearLayer::new(
            input_norm,
            post_attention_norm,
            *qkv,
            *z,
            *b,
            *a,
            *out_proj,
            delta_weights,
            delta_state,
            *gate_proj,
            *up_proj,
            *down_proj,
        );
        self.decode_linear_layers(state, &[layer], epsilon)
    }

    /// Executes one complete full-attention transformer layer without moving
    /// activations back to CPU. The Q8 KV cache remains owned by the request,
    /// while this method reuses the decode scratch stream across layers.
    pub fn decode_full_layer(
        &self,
        state: &mut MetalDecodeState,
        layer: &MetalDecodeFullLayer<'_>,
        kv_state: &mut Q8KvState,
        epsilon: f32,
    ) -> Result<(), MetalRuntimeError> {
        let command_buffer = self.command_queue.new_command_buffer();
        let sequence_length =
            self.encode_decode_full_layer(command_buffer, state, layer, kv_state, epsilon)?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        kv_state.sequence_length = sequence_length;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_decode_full_layer(
        &self,
        command_buffer: &metal::CommandBufferRef,
        state: &mut MetalDecodeState,
        layer: &MetalDecodeFullLayer<'_>,
        kv_state: &mut Q8KvState,
        epsilon: f32,
    ) -> Result<usize, MetalRuntimeError> {
        let config = layer.gqa.as_u32()?;
        if layer.gqa.kv_heads != kv_state.kv_heads || layer.gqa.head_dim != kv_state.head_dim {
            return Err(MetalRuntimeError::InvalidAttentionShape);
        }
        let hidden_elements = state.hidden_elements;
        if layer.input_norm.elements != hidden_elements {
            return Err(MetalRuntimeError::WrongLength {
                name: "decode full-attention input norm weights",
                actual: layer.input_norm.elements,
                expected: hidden_elements,
            });
        }
        if layer.post_attention_norm.elements != hidden_elements {
            return Err(MetalRuntimeError::WrongLength {
                name: "decode full-attention post-attention norm weights",
                actual: layer.post_attention_norm.elements,
                expected: hidden_elements,
            });
        }
        if layer.q_norm.elements != layer.gqa.head_dim {
            return Err(MetalRuntimeError::WrongLength {
                name: "decode full-attention query norm weights",
                actual: layer.q_norm.elements,
                expected: layer.gqa.head_dim,
            });
        }
        if layer.k_norm.elements != layer.gqa.head_dim {
            return Err(MetalRuntimeError::WrongLength {
                name: "decode full-attention key norm weights",
                actual: layer.k_norm.elements,
                expected: layer.gqa.head_dim,
            });
        }

        let query_elements = layer.gqa.num_heads.checked_mul(layer.gqa.head_dim).ok_or(
            MetalRuntimeError::DimensionOverflow("decode full-attention query elements"),
        )?;
        let q_with_gate_elements =
            query_elements
                .checked_mul(2)
                .ok_or(MetalRuntimeError::DimensionOverflow(
                    "decode full-attention query and gate elements",
                ))?;
        let kv_elements = layer.gqa.kv_heads.checked_mul(layer.gqa.head_dim).ok_or(
            MetalRuntimeError::DimensionOverflow("decode full-attention KV elements"),
        )?;
        let q_words = validate_decode_q4_job(
            hidden_elements,
            &layer.q_proj,
            q_with_gate_elements,
            "full-attention query projection",
        )?;
        let k_words = validate_decode_q4_job(
            hidden_elements,
            &layer.k_proj,
            kv_elements,
            "full-attention key projection",
        )?;
        let v_words = validate_decode_q4_job(
            hidden_elements,
            &layer.v_proj,
            kv_elements,
            "full-attention value projection",
        )?;
        let o_words = validate_decode_q4_job(
            query_elements,
            &layer.o_proj,
            hidden_elements,
            "full-attention output projection",
        )?;
        let gate_rows = layer.gate_proj.output_rows;
        let gate_words = validate_decode_q4_job(
            hidden_elements,
            &layer.gate_proj,
            gate_rows,
            "full-attention MLP gate projection",
        )?;
        let up_words = validate_decode_q4_job(
            hidden_elements,
            &layer.up_proj,
            gate_rows,
            "full-attention MLP up projection",
        )?;
        let down_words = validate_decode_q4_job(
            gate_rows,
            &layer.down_proj,
            hidden_elements,
            "full-attention MLP down projection",
        )?;
        let hidden_elements_u32 = u32::try_from(hidden_elements)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("decode hidden elements"))?;
        let gate_elements_u32 = u32::try_from(gate_rows)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("decode MLP elements"))?;

        let sequence_length =
            kv_state
                .sequence_length
                .checked_add(1)
                .ok_or(MetalRuntimeError::DimensionOverflow(
                    "full-attention Q8 KV sequence length",
                ))?;
        self.ensure_q8_kv_capacity(kv_state, sequence_length)?;
        let sequence_length_u32 = u32::try_from(sequence_length).map_err(|_| {
            MetalRuntimeError::DimensionOverflow("full-attention Q8 KV sequence length")
        })?;
        let token_index_u32 = u32::try_from(kv_state.sequence_length)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("full-attention Q8 KV token"))?;
        let score_elements = layer.gqa.num_heads.checked_mul(sequence_length).ok_or(
            MetalRuntimeError::DimensionOverflow("full-attention score elements"),
        )?;

        ensure_shared_buffer(
            &self.device,
            &mut state.qkv,
            checked_byte_len::<f32>(q_with_gate_elements)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.z,
            checked_byte_len::<f32>(query_elements)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.b,
            checked_byte_len::<f32>(query_elements)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.a,
            checked_byte_len::<f32>(kv_elements)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.convolved,
            checked_byte_len::<f32>(kv_elements)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.delta_output,
            checked_byte_len::<f32>(query_elements)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.gate,
            checked_byte_len::<f32>(gate_rows)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.up,
            checked_byte_len::<f32>(gate_rows)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.swiglu,
            checked_byte_len::<f32>(gate_rows)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.scores,
            checked_byte_len::<f32>(score_elements)?,
        )?;

        let hidden = &state.hidden.buffer;
        let normalized = &state.normalized.buffer;
        let post_norm = &state.post_norm.buffer;
        let mixed = &state.mixed.buffer;
        let q_with_gate = &state
            .qkv
            .as_ref()
            .expect("decode full-attention query buffer is initialized")
            .buffer;
        let query = &state
            .z
            .as_ref()
            .expect("decode full-attention normalized query buffer is initialized")
            .buffer;
        let gate = &state
            .b
            .as_ref()
            .expect("decode full-attention gate buffer is initialized")
            .buffer;
        let raw_key = &state
            .a
            .as_ref()
            .expect("decode full-attention key buffer is initialized")
            .buffer;
        let value = &state
            .convolved
            .as_ref()
            .expect("decode full-attention value buffer is initialized")
            .buffer;
        // This buffer first holds the normalized key. Once KV append has
        // consumed it, the attention-value kernel overwrites it with output.
        let attention_output = &state
            .delta_output
            .as_ref()
            .expect("decode full-attention output buffer is initialized")
            .buffer;
        let gate_output = &state
            .gate
            .as_ref()
            .expect("decode MLP gate buffer is initialized")
            .buffer;
        let up_output = &state
            .up
            .as_ref()
            .expect("decode MLP up buffer is initialized")
            .buffer;
        let swiglu_output = &state
            .swiglu
            .as_ref()
            .expect("decode SwiGLU buffer is initialized")
            .buffer;
        let scores = &state
            .scores
            .as_ref()
            .expect("decode full-attention score buffer is initialized")
            .buffer;

        let projection_encoder = command_buffer.new_compute_command_encoder();
        self.encode_rms_norm(
            projection_encoder,
            hidden,
            layer.input_norm,
            normalized,
            hidden_elements_u32,
            epsilon,
        );
        self.encode_q4_affine_matvec(
            projection_encoder,
            normalized,
            q_with_gate,
            &layer.q_proj,
            q_words,
        )?;
        self.encode_q4_affine_matvec(
            projection_encoder,
            normalized,
            raw_key,
            &layer.k_proj,
            k_words,
        )?;
        self.encode_q4_affine_matvec(
            projection_encoder,
            normalized,
            value,
            &layer.v_proj,
            v_words,
        )?;
        projection_encoder.end_encoding();

        let prepare_encoder = command_buffer.new_compute_command_encoder();
        self.encode_gqa_prepare_query(
            prepare_encoder,
            q_with_gate,
            layer.q_norm,
            query,
            gate,
            config,
            epsilon,
        );
        self.encode_gqa_prepare_key(
            prepare_encoder,
            raw_key,
            layer.k_norm,
            attention_output,
            config,
            epsilon,
        );
        prepare_encoder.end_encoding();

        let append_encoder = command_buffer.new_compute_command_encoder();
        append_encoder.set_compute_pipeline_state(&self.q8_kv_append);
        append_encoder.set_buffer(0, Some(attention_output), 0);
        append_encoder.set_buffer(1, Some(value), 0);
        append_encoder.set_buffer(2, Some(&kv_state.keys), 0);
        append_encoder.set_buffer(3, Some(&kv_state.values), 0);
        append_encoder.set_buffer(4, Some(&kv_state.key_scales), 0);
        append_encoder.set_buffer(5, Some(&kv_state.value_scales), 0);
        append_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            (&config.kv_heads as *const u32).cast(),
        );
        append_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            (&config.head_dim as *const u32).cast(),
        );
        append_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            (&token_index_u32 as *const u32).cast(),
        );
        append_encoder
            .set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        append_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(config.kv_heads), 2, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        append_encoder.end_encoding();

        let score_encoder = command_buffer.new_compute_command_encoder();
        score_encoder.set_compute_pipeline_state(&self.gqa_q8_scores);
        score_encoder.set_buffer(0, Some(query), 0);
        score_encoder.set_buffer(1, Some(&kv_state.keys), 0);
        score_encoder.set_buffer(2, Some(&kv_state.key_scales), 0);
        score_encoder.set_buffer(3, Some(scores), 0);
        score_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&sequence_length_u32 as *const u32).cast(),
        );
        score_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&config.num_heads as *const u32).cast(),
        );
        score_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            (&config.kv_heads as *const u32).cast(),
        );
        score_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            (&config.head_dim as *const u32).cast(),
        );
        score_encoder
            .set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        score_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(config.num_heads), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        score_encoder.end_encoding();

        let output_encoder = command_buffer.new_compute_command_encoder();
        output_encoder.set_compute_pipeline_state(&self.gqa_q8_values);
        output_encoder.set_buffer(0, Some(scores), 0);
        output_encoder.set_buffer(1, Some(&kv_state.values), 0);
        output_encoder.set_buffer(2, Some(&kv_state.value_scales), 0);
        output_encoder.set_buffer(3, Some(gate), 0);
        output_encoder.set_buffer(4, Some(attention_output), 0);
        output_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&sequence_length_u32 as *const u32).cast(),
        );
        output_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            (&config.num_heads as *const u32).cast(),
        );
        output_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            (&config.kv_heads as *const u32).cast(),
        );
        output_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            (&config.head_dim as *const u32).cast(),
        );
        output_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(config.num_heads), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        self.encode_q4_affine_matvec(
            output_encoder,
            attention_output,
            mixed,
            &layer.o_proj,
            o_words,
        )?;
        self.encode_add_in_place(output_encoder, hidden, mixed, hidden_elements_u32);
        self.encode_rms_norm(
            output_encoder,
            hidden,
            layer.post_attention_norm,
            post_norm,
            hidden_elements_u32,
            epsilon,
        );
        self.encode_q4_affine_matvec(
            output_encoder,
            post_norm,
            gate_output,
            &layer.gate_proj,
            gate_words,
        )?;
        self.encode_q4_affine_matvec(
            output_encoder,
            post_norm,
            up_output,
            &layer.up_proj,
            up_words,
        )?;
        self.encode_swiglu(
            output_encoder,
            gate_output,
            up_output,
            swiglu_output,
            gate_elements_u32,
        );
        self.encode_q4_affine_matvec(
            output_encoder,
            swiglu_output,
            mixed,
            &layer.down_proj,
            down_words,
        )?;
        self.encode_add_in_place(output_encoder, hidden, mixed, hidden_elements_u32);
        output_encoder.end_encoding();

        Ok(sequence_length)
    }

    /// Encodes a complete mixed DeltaNet/GQA transformer step in one command
    /// buffer. This preserves all GPU-side dependencies while avoiding a CPU
    /// submission and completion wait at every full-attention boundary.
    pub fn decode_layers(
        &self,
        state: &mut MetalDecodeState,
        layers: &mut [MetalDecodeLayer<'_>],
        epsilon: f32,
    ) -> Result<(), MetalRuntimeError> {
        if layers.is_empty() {
            return Err(MetalRuntimeError::EmptyDimension);
        }
        let command_buffer = self.command_queue.new_command_buffer();
        let mut full_sequence_lengths = Vec::new();
        for descriptor in layers.iter_mut() {
            match descriptor {
                MetalDecodeLayer::Linear(layer) => {
                    self.encode_decode_linear_layer(
                        command_buffer,
                        state,
                        layer.input_norm,
                        layer.post_attention_norm,
                        &layer.qkv,
                        &layer.z,
                        &layer.b,
                        &layer.a,
                        &layer.out_proj,
                        layer.delta_weights,
                        layer.delta_state,
                        &layer.gate_proj,
                        &layer.up_proj,
                        &layer.down_proj,
                        epsilon,
                    )?;
                }
                MetalDecodeLayer::Full(layer, kv_state) => {
                    full_sequence_lengths.push(self.encode_decode_full_layer(
                        command_buffer,
                        state,
                        layer,
                        kv_state,
                        epsilon,
                    )?);
                }
            }
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        let mut full_index = 0;
        for descriptor in layers {
            if let MetalDecodeLayer::Full(_, kv_state) = descriptor {
                kv_state.sequence_length = full_sequence_lengths[full_index];
                full_index += 1;
            }
        }
        Ok(())
    }

    /// Encodes consecutive DeltaNet layers into one command buffer. Each
    /// layer keeps the same causal recurrent state, while the residual stream
    /// and scratch activations never leave the GPU between layers.
    pub fn decode_linear_layers(
        &self,
        state: &mut MetalDecodeState,
        layers: &[MetalDecodeLinearLayer<'_>],
        epsilon: f32,
    ) -> Result<(), MetalRuntimeError> {
        if layers.is_empty() {
            return Err(MetalRuntimeError::EmptyDimension);
        }
        let command_buffer = self.command_queue.new_command_buffer();
        for layer in layers {
            self.encode_decode_linear_layer(
                command_buffer,
                state,
                layer.input_norm,
                layer.post_attention_norm,
                &layer.qkv,
                &layer.z,
                &layer.b,
                &layer.a,
                &layer.out_proj,
                layer.delta_weights,
                layer.delta_state,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
                epsilon,
            )?;
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_decode_linear_layer(
        &self,
        command_buffer: &metal::CommandBufferRef,
        state: &mut MetalDecodeState,
        input_norm: &MetalF32Buffer,
        post_attention_norm: &MetalF32Buffer,
        qkv: &MappedQ4AffineJob<'_>,
        z: &MappedQ4AffineJob<'_>,
        b: &MappedQ4AffineJob<'_>,
        a: &MappedQ4AffineJob<'_>,
        out_proj: &MappedQ4AffineJob<'_>,
        delta_weights: &MetalDeltaNetWeights,
        delta_state: &MetalDeltaNetState,
        gate_proj: &MappedQ4AffineJob<'_>,
        up_proj: &MappedQ4AffineJob<'_>,
        down_proj: &MappedQ4AffineJob<'_>,
        epsilon: f32,
    ) -> Result<(), MetalRuntimeError> {
        let hidden_elements = state.hidden_elements;
        if input_norm.elements != hidden_elements {
            return Err(MetalRuntimeError::WrongLength {
                name: "decode input norm weights",
                actual: input_norm.elements,
                expected: hidden_elements,
            });
        }
        if post_attention_norm.elements != hidden_elements {
            return Err(MetalRuntimeError::WrongLength {
                name: "decode post-attention norm weights",
                actual: post_attention_norm.elements,
                expected: hidden_elements,
            });
        }

        let config = delta_weights.config;
        validate_deltanet_config(config)?;
        let channels = config.channels()?;
        let value_elements = config
            .value_heads
            .checked_mul(config.value_head_dim)
            .ok_or(MetalRuntimeError::DimensionOverflow(
                "DeltaNet value elements",
            ))?;
        let qkv_words = validate_decode_q4_job(hidden_elements, qkv, channels, "DeltaNet qkv")?;
        let z_words = validate_decode_q4_job(hidden_elements, z, value_elements, "DeltaNet z")?;
        let b_words = validate_decode_q4_job(hidden_elements, b, config.value_heads, "DeltaNet b")?;
        let a_words = validate_decode_q4_job(hidden_elements, a, config.value_heads, "DeltaNet a")?;
        let out_words = validate_decode_q4_job(
            value_elements,
            out_proj,
            hidden_elements,
            "DeltaNet out projection",
        )?;
        let gate_rows = gate_proj.output_rows;
        let gate_words =
            validate_decode_q4_job(hidden_elements, gate_proj, gate_rows, "MLP gate projection")?;
        let up_words =
            validate_decode_q4_job(hidden_elements, up_proj, gate_rows, "MLP up projection")?;
        let down_words =
            validate_decode_q4_job(gate_rows, down_proj, hidden_elements, "MLP down projection")?;
        let hidden_elements_u32 = u32::try_from(hidden_elements)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("decode hidden elements"))?;
        let gate_elements_u32 = u32::try_from(gate_rows)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("decode MLP elements"))?;

        ensure_shared_buffer(
            &self.device,
            &mut state.qkv,
            checked_byte_len::<f32>(channels)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.z,
            checked_byte_len::<f32>(value_elements)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.b,
            checked_byte_len::<f32>(config.value_heads)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.a,
            checked_byte_len::<f32>(config.value_heads)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.convolved,
            checked_byte_len::<f32>(channels)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.delta_output,
            checked_byte_len::<f32>(value_elements)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.gate,
            checked_byte_len::<f32>(gate_rows)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.up,
            checked_byte_len::<f32>(gate_rows)?,
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut state.swiglu,
            checked_byte_len::<f32>(gate_rows)?,
        )?;

        let hidden = &state.hidden.buffer;
        let normalized = &state.normalized.buffer;
        let post_norm = &state.post_norm.buffer;
        let mixed = &state.mixed.buffer;
        let qkv_output = &state
            .qkv
            .as_ref()
            .expect("decode qkv buffer is initialized")
            .buffer;
        let z_output = &state
            .z
            .as_ref()
            .expect("decode z buffer is initialized")
            .buffer;
        let b_output = &state
            .b
            .as_ref()
            .expect("decode b buffer is initialized")
            .buffer;
        let a_output = &state
            .a
            .as_ref()
            .expect("decode a buffer is initialized")
            .buffer;
        let convolved = &state
            .convolved
            .as_ref()
            .expect("decode convolved buffer is initialized")
            .buffer;
        let delta_output = &state
            .delta_output
            .as_ref()
            .expect("decode DeltaNet output buffer is initialized")
            .buffer;
        let gate_output = &state
            .gate
            .as_ref()
            .expect("decode gate buffer is initialized")
            .buffer;
        let up_output = &state
            .up
            .as_ref()
            .expect("decode up buffer is initialized")
            .buffer;
        let swiglu_output = &state
            .swiglu
            .as_ref()
            .expect("decode SwiGLU buffer is initialized")
            .buffer;

        let projection_encoder = command_buffer.new_compute_command_encoder();
        self.encode_rms_norm(
            projection_encoder,
            hidden,
            input_norm,
            normalized,
            hidden_elements_u32,
            epsilon,
        );
        self.encode_q4_affine_matvec(projection_encoder, normalized, qkv_output, qkv, qkv_words)?;
        self.encode_q4_affine_matvec(projection_encoder, normalized, z_output, z, z_words)?;
        self.encode_q4_affine_matvec(projection_encoder, normalized, b_output, b, b_words)?;
        self.encode_q4_affine_matvec(projection_encoder, normalized, a_output, a, a_words)?;
        projection_encoder.end_encoding();

        self.encode_deltanet_step(
            command_buffer,
            delta_weights,
            delta_state,
            qkv_output,
            z_output,
            b_output,
            a_output,
            convolved,
            delta_output,
            epsilon,
        )?;

        let output_encoder = command_buffer.new_compute_command_encoder();
        self.encode_q4_affine_matvec(output_encoder, delta_output, mixed, out_proj, out_words)?;
        self.encode_add_in_place(output_encoder, hidden, mixed, hidden_elements_u32);
        self.encode_rms_norm(
            output_encoder,
            hidden,
            post_attention_norm,
            post_norm,
            hidden_elements_u32,
            epsilon,
        );
        self.encode_q4_affine_matvec(
            output_encoder,
            post_norm,
            gate_output,
            gate_proj,
            gate_words,
        )?;
        self.encode_q4_affine_matvec(output_encoder, post_norm, up_output, up_proj, up_words)?;
        self.encode_swiglu(
            output_encoder,
            gate_output,
            up_output,
            swiglu_output,
            gate_elements_u32,
        );
        self.encode_q4_affine_matvec(output_encoder, swiglu_output, mixed, down_proj, down_words)?;
        self.encode_add_in_place(output_encoder, hidden, mixed, hidden_elements_u32);
        output_encoder.end_encoding();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn q4_affine_matvec_mapped(
        &self,
        input: &[f32],
        weights: &metal::Buffer,
        weight_offset: u64,
        scales: &metal::Buffer,
        scale_offset: u64,
        biases: &metal::Buffer,
        bias_offset: u64,
        output_rows: usize,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        let words_per_row = validate_mapped_q4_affine_matvec(
            input,
            weights,
            weight_offset,
            scales,
            scale_offset,
            biases,
            bias_offset,
            output_rows,
        )?;

        let input_buffer = self.buffer_from_slice(input)?;
        self.dispatch_q4_affine_matvec(
            &input_buffer,
            weights,
            weight_offset,
            scales,
            scale_offset,
            biases,
            bias_offset,
            output_rows,
            words_per_row,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn q4_affine_matvec_mapped_unaligned(
        &self,
        input: &[f32],
        weights: &metal::Buffer,
        weight_offset: u64,
        scales: &metal::Buffer,
        scale_offset: u64,
        biases: &metal::Buffer,
        bias_offset: u64,
        output_rows: usize,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        let words_per_row = validate_mapped_q4_affine_matvec(
            input,
            weights,
            weight_offset,
            scales,
            scale_offset,
            biases,
            bias_offset,
            output_rows,
        )?;
        let input_buffer = self.buffer_from_slice(input)?;
        self.dispatch_q4_affine_matvec_unaligned(
            &input_buffer,
            weights,
            weight_offset,
            scales,
            scale_offset,
            biases,
            bias_offset,
            output_rows,
            words_per_row,
        )
    }

    /// Runs several mapped affine-Q4 projections over the same activation in
    /// one command buffer. The activation and result buffers are retained by
    /// the runtime, so steady-state decode avoids per-projection Metal buffer
    /// allocation in addition to command-buffer and fence overhead.
    pub fn q4_affine_matvec_mapped_batch(
        &self,
        input: &[f32],
        jobs: &[MappedQ4AffineJob<'_>],
    ) -> Result<Vec<Vec<f32>>, MetalRuntimeError> {
        if jobs.is_empty() {
            return Err(MetalRuntimeError::EmptyDimension);
        }
        let words_per_rows: Vec<u32> = jobs
            .iter()
            .map(|job| {
                validate_mapped_q4_affine_matvec(
                    input,
                    job.weights,
                    job.weight_offset,
                    job.scales,
                    job.scale_offset,
                    job.biases,
                    job.bias_offset,
                    job.output_rows,
                )
                .and_then(|words| {
                    u32::try_from(words)
                        .map_err(|_| MetalRuntimeError::DimensionOverflow("words per row"))
                })
            })
            .collect::<Result<_, _>>()?;
        let input_bytes = checked_byte_len::<f32>(input.len())?;
        let mut activations = self
            .q4_activations
            .lock()
            .map_err(|_| MetalRuntimeError::ActivationPoolPoisoned)?;
        ensure_shared_buffer(&self.device, &mut activations.input, input_bytes)?;
        if activations.outputs.len() < jobs.len() {
            activations.outputs.resize_with(jobs.len(), || None);
        }
        for (slot, job) in activations.outputs.iter_mut().zip(jobs) {
            ensure_shared_buffer(
                &self.device,
                slot,
                checked_byte_len::<f32>(job.output_rows)?,
            )?;
        }
        let input_buffer = &activations
            .input
            .as_ref()
            .expect("input buffer is initialized")
            .buffer;
        copy_slice_to_buffer(input_buffer, input);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        for ((index, job), words_per_row) in jobs.iter().enumerate().zip(words_per_rows) {
            let output = &activations.outputs[index]
                .as_ref()
                .expect("output buffer is initialized")
                .buffer;
            self.encode_q4_affine_matvec(encoder, input_buffer, output, job, words_per_row)?;
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }

        jobs.iter()
            .enumerate()
            .map(|(index, job)| {
                let output = &activations.outputs[index]
                    .as_ref()
                    .expect("output buffer is initialized")
                    .buffer;
                Ok(unsafe {
                    std::slice::from_raw_parts(output.contents().cast::<f32>(), job.output_rows)
                        .to_vec()
                })
            })
            .collect()
    }

    /// Applies each mapped Q4 affine projection to every row of a prompt in
    /// one GPU submission. Results are row-major: batch row first, then the
    /// projection output row. This is the layer-major prefill primitive.
    pub fn q4_affine_matmul_mapped_batch(
        &self,
        input: &[f32],
        batch_size: usize,
        jobs: &[MappedQ4AffineJob<'_>],
    ) -> Result<Vec<Vec<f32>>, MetalRuntimeError> {
        if jobs.is_empty() || batch_size == 0 || input.is_empty() || input.len() % batch_size != 0 {
            return Err(MetalRuntimeError::EmptyDimension);
        }
        let input_elements = input.len() / batch_size;
        let words_per_rows = jobs
            .iter()
            .map(|job| {
                validate_mapped_q4_affine_shape(
                    input_elements,
                    job.weights,
                    job.weight_offset,
                    job.scales,
                    job.scale_offset,
                    job.biases,
                    job.bias_offset,
                    job.output_rows,
                )
                .and_then(|words| {
                    u32::try_from(words)
                        .map_err(|_| MetalRuntimeError::DimensionOverflow("words per row"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        u32::try_from(batch_size)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("Q4 batch size"))?;

        if self.should_use_mps_q4_prefill(batch_size, jobs) {
            return self.q4_affine_matmul_mapped_batch_mps(
                input,
                batch_size,
                jobs,
                &words_per_rows,
            );
        }

        let mut activations = self
            .q4_activations
            .lock()
            .map_err(|_| MetalRuntimeError::ActivationPoolPoisoned)?;
        ensure_shared_buffer(
            &self.device,
            &mut activations.input,
            checked_byte_len::<f32>(input.len())?,
        )?;
        if activations.outputs.len() < jobs.len() {
            activations.outputs.resize_with(jobs.len(), || None);
        }
        for (slot, job) in activations.outputs.iter_mut().zip(jobs) {
            let output_elements = batch_size.checked_mul(job.output_rows).ok_or(
                MetalRuntimeError::DimensionOverflow("Q4 batch output elements"),
            )?;
            ensure_shared_buffer(
                &self.device,
                slot,
                checked_byte_len::<f32>(output_elements)?,
            )?;
        }

        let input_buffer = &activations
            .input
            .as_ref()
            .expect("input buffer is initialized")
            .buffer;
        copy_slice_to_buffer(input_buffer, input);
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        for ((index, job), words_per_row) in jobs.iter().enumerate().zip(words_per_rows) {
            let output = &activations.outputs[index]
                .as_ref()
                .expect("output buffer is initialized")
                .buffer;
            self.encode_q4_affine_matmul(
                encoder,
                input_buffer,
                output,
                job,
                words_per_row,
                batch_size,
            )?;
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }

        jobs.iter()
            .enumerate()
            .map(|(index, job)| {
                let output = &activations.outputs[index]
                    .as_ref()
                    .expect("output buffer is initialized")
                    .buffer;
                let output_elements = batch_size.checked_mul(job.output_rows).ok_or(
                    MetalRuntimeError::DimensionOverflow("Q4 batch output elements"),
                )?;
                Ok(unsafe {
                    std::slice::from_raw_parts(output.contents().cast::<f32>(), output_elements)
                        .to_vec()
                })
            })
            .collect()
    }

    fn should_use_mps_q4_prefill(&self, batch_size: usize, jobs: &[MappedQ4AffineJob<'_>]) -> bool {
        self.mps_q4_prefill && batch_size >= Q4_MPS_PREFILL_MIN_BATCH && !jobs.is_empty()
    }

    fn q4_affine_matmul_mapped_batch_mps(
        &self,
        input: &[f32],
        batch_size: usize,
        jobs: &[MappedQ4AffineJob<'_>],
        words_per_rows: &[u32],
    ) -> Result<Vec<Vec<f32>>, MetalRuntimeError> {
        let input_elements = input.len() / batch_size;
        let mut activations = self
            .q4_activations
            .lock()
            .map_err(|_| MetalRuntimeError::ActivationPoolPoisoned)?;
        ensure_shared_buffer(
            &self.device,
            &mut activations.input,
            checked_byte_len::<f32>(input.len())?,
        )?;
        ensure_private_buffer(
            &self.device,
            &mut activations.input_half,
            checked_byte_len::<u16>(input.len())?,
        )?;
        let largest_weight_elements = jobs.iter().try_fold(0_usize, |largest, job| {
            input_elements
                .checked_mul(job.output_rows)
                .map(|elements| largest.max(elements))
                .ok_or(MetalRuntimeError::DimensionOverflow(
                    "Q4 weight scratch elements",
                ))
        })?;
        ensure_private_buffer(
            &self.device,
            &mut activations.weights_half,
            checked_byte_len::<u16>(largest_weight_elements)?,
        )?;
        if activations.half_slots.is_empty() {
            activations.half_slots.push(None);
        }
        let largest_output_elements = jobs.iter().try_fold(0_usize, |largest, job| {
            batch_size
                .checked_mul(job.output_rows)
                .map(|elements| largest.max(elements))
                .ok_or(MetalRuntimeError::DimensionOverflow(
                    "Q4 output scratch elements",
                ))
        })?;
        ensure_private_buffer(
            &self.device,
            &mut activations.half_slots[0],
            checked_byte_len::<u16>(largest_output_elements)?,
        )?;
        if activations.outputs.len() < jobs.len() {
            activations.outputs.resize_with(jobs.len(), || None);
        }
        for (slot, job) in activations.outputs.iter_mut().zip(jobs) {
            let output_elements = batch_size.checked_mul(job.output_rows).ok_or(
                MetalRuntimeError::DimensionOverflow("Q4 batch output elements"),
            )?;
            ensure_shared_buffer(
                &self.device,
                slot,
                checked_byte_len::<f32>(output_elements)?,
            )?;
        }

        let input_buffer = &activations
            .input
            .as_ref()
            .expect("input buffer is initialized")
            .buffer;
        let input_half = &activations
            .input_half
            .as_ref()
            .expect("half input buffer is initialized")
            .buffer;
        let weights_half = &activations
            .weights_half
            .as_ref()
            .expect("half weight buffer is initialized")
            .buffer;
        let projection_half = &activations.half_slots[0]
            .as_ref()
            .expect("half projection buffer is initialized")
            .buffer;
        copy_slice_to_buffer(input_buffer, input);

        let command_buffer = self.command_queue.new_command_buffer();
        let conversion_encoder = command_buffer.new_compute_command_encoder();
        self.encode_f32_to_f16(conversion_encoder, input_buffer, input_half, input.len())?;
        conversion_encoder.end_encoding();

        let input_matrix = MpsMatrix::new_fp16(input_half, batch_size, input_elements)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let mut matrices = Vec::with_capacity(jobs.len() * 2);
        let mut gemms = Vec::with_capacity(jobs.len());
        for ((index, job), words_per_row) in jobs.iter().enumerate().zip(words_per_rows.iter()) {
            let output = &activations.outputs[index]
                .as_ref()
                .expect("output buffer is initialized")
                .buffer;
            let dequantize_encoder = command_buffer.new_compute_command_encoder();
            self.encode_q4_affine_dequantize_f16(
                dequantize_encoder,
                weights_half,
                0,
                job,
                *words_per_row,
            )?;
            dequantize_encoder.end_encoding();

            let right_matrix = MpsMatrix::new_fp16(weights_half, job.output_rows, input_elements)
                .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
            let result_matrix = MpsMatrix::new_fp16(projection_half, batch_size, job.output_rows)
                .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
            let gemm = MpsFp16Gemm::new(&self.device, batch_size, job.output_rows, input_elements)
                .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
            gemm.encode(command_buffer, &input_matrix, &right_matrix, &result_matrix);

            let output_elements = batch_size.checked_mul(job.output_rows).ok_or(
                MetalRuntimeError::DimensionOverflow("Q4 batch output elements"),
            )?;
            let conversion_encoder = command_buffer.new_compute_command_encoder();
            self.encode_f16_to_f32(conversion_encoder, projection_half, output, output_elements)?;
            conversion_encoder.end_encoding();
            matrices.push(right_matrix);
            matrices.push(result_matrix);
            gemms.push(gemm);
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }

        jobs.iter()
            .enumerate()
            .map(|(index, job)| {
                let output = &activations.outputs[index]
                    .as_ref()
                    .expect("output buffer is initialized")
                    .buffer;
                let output_elements = batch_size.checked_mul(job.output_rows).ok_or(
                    MetalRuntimeError::DimensionOverflow("Q4 batch output elements"),
                )?;
                Ok(unsafe {
                    std::slice::from_raw_parts(output.contents().cast::<f32>(), output_elements)
                        .to_vec()
                })
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn q4_affine_mlp_mapped_batch_mps(
        &self,
        input: &[f32],
        batch_size: usize,
        gate: &MappedQ4AffineJob<'_>,
        up: &MappedQ4AffineJob<'_>,
        down: &MappedQ4AffineJob<'_>,
        gate_words: u32,
        up_words: u32,
        down_words: u32,
        intermediate_elements: usize,
        output_elements: usize,
        swiglu_elements: u32,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        if self.mps_q4_mlp_fusion {
            return self.q4_affine_mlp_mapped_batch_mps_fused_gate_up(
                input,
                batch_size,
                gate,
                up,
                down,
                gate_words,
                up_words,
                down_words,
                intermediate_elements,
                output_elements,
                swiglu_elements,
            );
        }
        let input_elements = input.len() / batch_size;
        let gate_weight_elements = input_elements.checked_mul(gate.output_rows).ok_or(
            MetalRuntimeError::DimensionOverflow("MLP gate weight scratch elements"),
        )?;
        let up_weight_elements = input_elements.checked_mul(up.output_rows).ok_or(
            MetalRuntimeError::DimensionOverflow("MLP up weight scratch elements"),
        )?;
        let down_weight_elements = gate.output_rows.checked_mul(down.output_rows).ok_or(
            MetalRuntimeError::DimensionOverflow("MLP down weight scratch elements"),
        )?;
        let largest_weight_elements = gate_weight_elements
            .max(up_weight_elements)
            .max(down_weight_elements);
        let gate_or_down_elements = intermediate_elements.max(output_elements);

        let mut activations = self
            .q4_activations
            .lock()
            .map_err(|_| MetalRuntimeError::ActivationPoolPoisoned)?;
        ensure_shared_buffer(
            &self.device,
            &mut activations.input,
            checked_byte_len::<f32>(input.len())?,
        )?;
        ensure_private_buffer(
            &self.device,
            &mut activations.input_half,
            checked_byte_len::<u16>(input.len())?,
        )?;
        ensure_private_buffer(
            &self.device,
            &mut activations.weights_half,
            checked_byte_len::<u16>(largest_weight_elements)?,
        )?;
        if activations.half_slots.len() < 3 {
            activations.half_slots.resize_with(3, || None);
        }
        for (slot, elements) in activations.half_slots.iter_mut().take(3).zip([
            gate_or_down_elements,
            intermediate_elements,
            intermediate_elements,
        ]) {
            ensure_private_buffer(&self.device, slot, checked_byte_len::<u16>(elements)?)?;
        }
        if activations.outputs.is_empty() {
            activations.outputs.push(None);
        }
        ensure_shared_buffer(
            &self.device,
            &mut activations.outputs[0],
            checked_byte_len::<f32>(output_elements)?,
        )?;

        let input_buffer = &activations
            .input
            .as_ref()
            .expect("input buffer is initialized")
            .buffer;
        let input_half = &activations
            .input_half
            .as_ref()
            .expect("half input buffer is initialized")
            .buffer;
        let weights_half = &activations
            .weights_half
            .as_ref()
            .expect("half weight buffer is initialized")
            .buffer;
        let gate_or_down_half = &activations.half_slots[0]
            .as_ref()
            .expect("half gate buffer is initialized")
            .buffer;
        let up_half = &activations.half_slots[1]
            .as_ref()
            .expect("half up buffer is initialized")
            .buffer;
        let swiglu_half = &activations.half_slots[2]
            .as_ref()
            .expect("half SwiGLU buffer is initialized")
            .buffer;
        let output = &activations.outputs[0]
            .as_ref()
            .expect("output buffer is initialized")
            .buffer;
        copy_slice_to_buffer(input_buffer, input);

        let command_buffer = self.command_queue.new_command_buffer();
        let conversion_encoder = command_buffer.new_compute_command_encoder();
        self.encode_f32_to_f16(conversion_encoder, input_buffer, input_half, input.len())?;
        conversion_encoder.end_encoding();
        let input_matrix = MpsMatrix::new_fp16(input_half, batch_size, input_elements)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;

        let dequantize_encoder = command_buffer.new_compute_command_encoder();
        self.encode_q4_affine_dequantize_f16(
            dequantize_encoder,
            weights_half,
            0,
            gate,
            gate_words,
        )?;
        dequantize_encoder.end_encoding();
        let gate_weight = MpsMatrix::new_fp16(weights_half, gate.output_rows, input_elements)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let gate_result = MpsMatrix::new_fp16(gate_or_down_half, batch_size, gate.output_rows)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let gate_gemm =
            MpsFp16Gemm::new(&self.device, batch_size, gate.output_rows, input_elements)
                .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        gate_gemm.encode(command_buffer, &input_matrix, &gate_weight, &gate_result);

        let dequantize_encoder = command_buffer.new_compute_command_encoder();
        self.encode_q4_affine_dequantize_f16(dequantize_encoder, weights_half, 0, up, up_words)?;
        dequantize_encoder.end_encoding();
        let up_weight = MpsMatrix::new_fp16(weights_half, up.output_rows, input_elements)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let up_result = MpsMatrix::new_fp16(up_half, batch_size, up.output_rows)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let up_gemm = MpsFp16Gemm::new(&self.device, batch_size, up.output_rows, input_elements)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        up_gemm.encode(command_buffer, &input_matrix, &up_weight, &up_result);

        let swiglu_encoder = command_buffer.new_compute_command_encoder();
        self.encode_swiglu_half(
            swiglu_encoder,
            gate_or_down_half,
            up_half,
            swiglu_half,
            swiglu_elements,
        );
        swiglu_encoder.end_encoding();

        let dequantize_encoder = command_buffer.new_compute_command_encoder();
        self.encode_q4_affine_dequantize_f16(
            dequantize_encoder,
            weights_half,
            0,
            down,
            down_words,
        )?;
        dequantize_encoder.end_encoding();
        let swiglu_matrix = MpsMatrix::new_fp16(swiglu_half, batch_size, gate.output_rows)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let down_weight = MpsMatrix::new_fp16(weights_half, down.output_rows, gate.output_rows)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let down_result = MpsMatrix::new_fp16(gate_or_down_half, batch_size, down.output_rows)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let down_gemm =
            MpsFp16Gemm::new(&self.device, batch_size, down.output_rows, gate.output_rows)
                .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        down_gemm.encode(command_buffer, &swiglu_matrix, &down_weight, &down_result);

        let conversion_encoder = command_buffer.new_compute_command_encoder();
        self.encode_f16_to_f32(
            conversion_encoder,
            gate_or_down_half,
            output,
            output_elements,
        )?;
        conversion_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        let result = unsafe {
            std::slice::from_raw_parts(output.contents().cast::<f32>(), output_elements).to_vec()
        };
        // MPS encoders retain their resources while the command buffer runs.
        // Keep all wrappers alive until completion even though the backing
        // buffers are deliberately reused in sequence.
        drop((
            input_matrix,
            gate_weight,
            gate_result,
            gate_gemm,
            up_weight,
            up_result,
            up_gemm,
            swiglu_matrix,
            down_weight,
            down_result,
            down_gemm,
        ));
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn q4_affine_mlp_mapped_batch_mps_fused_gate_up(
        &self,
        input: &[f32],
        batch_size: usize,
        gate: &MappedQ4AffineJob<'_>,
        up: &MappedQ4AffineJob<'_>,
        down: &MappedQ4AffineJob<'_>,
        gate_words: u32,
        up_words: u32,
        down_words: u32,
        intermediate_elements: usize,
        output_elements: usize,
        swiglu_elements: u32,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        let input_elements = input.len() / batch_size;
        let gate_weight_elements = input_elements.checked_mul(gate.output_rows).ok_or(
            MetalRuntimeError::DimensionOverflow("MLP gate weight scratch elements"),
        )?;
        let up_weight_elements = input_elements.checked_mul(up.output_rows).ok_or(
            MetalRuntimeError::DimensionOverflow("MLP up weight scratch elements"),
        )?;
        let gate_and_up_weight_elements =
            gate_weight_elements.checked_add(up_weight_elements).ok_or(
                MetalRuntimeError::DimensionOverflow("MLP gate/up weight scratch elements"),
            )?;
        let down_weight_elements = gate.output_rows.checked_mul(down.output_rows).ok_or(
            MetalRuntimeError::DimensionOverflow("MLP down weight scratch elements"),
        )?;
        let largest_weight_elements = gate_and_up_weight_elements.max(down_weight_elements);
        let gate_and_up_rows = gate.output_rows.checked_add(up.output_rows).ok_or(
            MetalRuntimeError::DimensionOverflow("MLP gate/up output rows"),
        )?;
        let gate_and_up_elements = batch_size.checked_mul(gate_and_up_rows).ok_or(
            MetalRuntimeError::DimensionOverflow("MLP gate/up output elements"),
        )?;
        let gate_up_or_down_elements = gate_and_up_elements.max(output_elements);
        let gate_weight_offset = checked_byte_len::<u16>(gate_weight_elements)?;
        let intermediate_width = u32::try_from(gate.output_rows)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("MLP intermediate width"))?;

        let mut activations = self
            .q4_activations
            .lock()
            .map_err(|_| MetalRuntimeError::ActivationPoolPoisoned)?;
        ensure_shared_buffer(
            &self.device,
            &mut activations.input,
            checked_byte_len::<f32>(input.len())?,
        )?;
        ensure_private_buffer(
            &self.device,
            &mut activations.input_half,
            checked_byte_len::<u16>(input.len())?,
        )?;
        ensure_private_buffer(
            &self.device,
            &mut activations.weights_half,
            checked_byte_len::<u16>(largest_weight_elements)?,
        )?;
        if activations.half_slots.len() < 2 {
            activations.half_slots.resize_with(2, || None);
        }
        ensure_private_buffer(
            &self.device,
            &mut activations.half_slots[0],
            checked_byte_len::<u16>(gate_up_or_down_elements)?,
        )?;
        ensure_private_buffer(
            &self.device,
            &mut activations.half_slots[1],
            checked_byte_len::<u16>(intermediate_elements)?,
        )?;
        if activations.outputs.is_empty() {
            activations.outputs.push(None);
        }
        ensure_shared_buffer(
            &self.device,
            &mut activations.outputs[0],
            checked_byte_len::<f32>(output_elements)?,
        )?;

        let input_buffer = &activations
            .input
            .as_ref()
            .expect("input buffer is initialized")
            .buffer;
        let input_half = &activations
            .input_half
            .as_ref()
            .expect("half input buffer is initialized")
            .buffer;
        let weights_half = &activations
            .weights_half
            .as_ref()
            .expect("half weight buffer is initialized")
            .buffer;
        let gate_and_up_or_down_half = &activations.half_slots[0]
            .as_ref()
            .expect("half gate/up buffer is initialized")
            .buffer;
        let swiglu_half = &activations.half_slots[1]
            .as_ref()
            .expect("half SwiGLU buffer is initialized")
            .buffer;
        let output = &activations.outputs[0]
            .as_ref()
            .expect("output buffer is initialized")
            .buffer;
        copy_slice_to_buffer(input_buffer, input);

        let command_buffer = self.command_queue.new_command_buffer();
        let conversion_encoder = command_buffer.new_compute_command_encoder();
        self.encode_f32_to_f16(conversion_encoder, input_buffer, input_half, input.len())?;
        conversion_encoder.end_encoding();
        let input_matrix = MpsMatrix::new_fp16(input_half, batch_size, input_elements)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;

        let dequantize_encoder = command_buffer.new_compute_command_encoder();
        self.encode_q4_affine_dequantize_f16(
            dequantize_encoder,
            weights_half,
            0,
            gate,
            gate_words,
        )?;
        self.encode_q4_affine_dequantize_f16(
            dequantize_encoder,
            weights_half,
            gate_weight_offset,
            up,
            up_words,
        )?;
        dequantize_encoder.end_encoding();
        let gate_and_up_weight =
            MpsMatrix::new_fp16(weights_half, gate_and_up_rows, input_elements)
                .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let gate_and_up_result =
            MpsMatrix::new_fp16(gate_and_up_or_down_half, batch_size, gate_and_up_rows)
                .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let gate_and_up_gemm =
            MpsFp16Gemm::new(&self.device, batch_size, gate_and_up_rows, input_elements)
                .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        gate_and_up_gemm.encode(
            command_buffer,
            &input_matrix,
            &gate_and_up_weight,
            &gate_and_up_result,
        );

        let swiglu_encoder = command_buffer.new_compute_command_encoder();
        self.encode_swiglu_half_split(
            swiglu_encoder,
            gate_and_up_or_down_half,
            swiglu_half,
            intermediate_width,
            swiglu_elements,
        );
        swiglu_encoder.end_encoding();

        let dequantize_encoder = command_buffer.new_compute_command_encoder();
        self.encode_q4_affine_dequantize_f16(
            dequantize_encoder,
            weights_half,
            0,
            down,
            down_words,
        )?;
        dequantize_encoder.end_encoding();
        let swiglu_matrix = MpsMatrix::new_fp16(swiglu_half, batch_size, gate.output_rows)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let down_weight = MpsMatrix::new_fp16(weights_half, down.output_rows, gate.output_rows)
            .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let down_result =
            MpsMatrix::new_fp16(gate_and_up_or_down_half, batch_size, down.output_rows)
                .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        let down_gemm =
            MpsFp16Gemm::new(&self.device, batch_size, down.output_rows, gate.output_rows)
                .map_err(|error| MetalRuntimeError::Mps(error.to_string()))?;
        down_gemm.encode(command_buffer, &swiglu_matrix, &down_weight, &down_result);

        let conversion_encoder = command_buffer.new_compute_command_encoder();
        self.encode_f16_to_f32(
            conversion_encoder,
            gate_and_up_or_down_half,
            output,
            output_elements,
        )?;
        conversion_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        let result = unsafe {
            std::slice::from_raw_parts(output.contents().cast::<f32>(), output_elements).to_vec()
        };
        drop((
            input_matrix,
            gate_and_up_weight,
            gate_and_up_result,
            gate_and_up_gemm,
            swiglu_matrix,
            down_weight,
            down_result,
            down_gemm,
        ));
        Ok(result)
    }

    /// Fuses the batched MLP command sequence: gate/up Q4 projections,
    /// SwiGLU, and down Q4 projection share GPU buffers and one completion
    /// fence. The caller supplies a post-normalized input matrix.
    pub fn q4_affine_mlp_mapped_batch(
        &self,
        input: &[f32],
        batch_size: usize,
        gate: &MappedQ4AffineJob<'_>,
        up: &MappedQ4AffineJob<'_>,
        down: &MappedQ4AffineJob<'_>,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        if batch_size == 0 || input.is_empty() || input.len() % batch_size != 0 {
            return Err(MetalRuntimeError::EmptyDimension);
        }
        let input_elements = input.len() / batch_size;
        let gate_words = validate_mapped_q4_affine_shape(
            input_elements,
            gate.weights,
            gate.weight_offset,
            gate.scales,
            gate.scale_offset,
            gate.biases,
            gate.bias_offset,
            gate.output_rows,
        )?;
        let up_words = validate_mapped_q4_affine_shape(
            input_elements,
            up.weights,
            up.weight_offset,
            up.scales,
            up.scale_offset,
            up.biases,
            up.bias_offset,
            up.output_rows,
        )?;
        if gate.output_rows != up.output_rows {
            return Err(MetalRuntimeError::WrongLength {
                name: "MLP gate/up output rows",
                actual: up.output_rows,
                expected: gate.output_rows,
            });
        }
        let down_words = validate_mapped_q4_affine_shape(
            gate.output_rows,
            down.weights,
            down.weight_offset,
            down.scales,
            down.scale_offset,
            down.biases,
            down.bias_offset,
            down.output_rows,
        )?;
        let gate_words = u32::try_from(gate_words)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("words per row"))?;
        let up_words = u32::try_from(up_words)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("words per row"))?;
        let down_words = u32::try_from(down_words)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("words per row"))?;
        u32::try_from(batch_size)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("Q4 batch size"))?;
        let intermediate_elements = batch_size.checked_mul(gate.output_rows).ok_or(
            MetalRuntimeError::DimensionOverflow("MLP intermediate elements"),
        )?;
        let output_elements = batch_size
            .checked_mul(down.output_rows)
            .ok_or(MetalRuntimeError::DimensionOverflow("MLP output elements"))?;
        let swiglu_elements = u32::try_from(intermediate_elements)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("MLP intermediate elements"))?;

        if self.mps_q4_prefill && batch_size >= Q4_MPS_PREFILL_MIN_BATCH {
            return self.q4_affine_mlp_mapped_batch_mps(
                input,
                batch_size,
                gate,
                up,
                down,
                gate_words,
                up_words,
                down_words,
                intermediate_elements,
                output_elements,
                swiglu_elements,
            );
        }

        let mut activations = self
            .q4_activations
            .lock()
            .map_err(|_| MetalRuntimeError::ActivationPoolPoisoned)?;
        ensure_shared_buffer(
            &self.device,
            &mut activations.input,
            checked_byte_len::<f32>(input.len())?,
        )?;
        if activations.outputs.len() < 4 {
            activations.outputs.resize_with(4, || None);
        }
        for (index, elements) in [
            intermediate_elements,
            intermediate_elements,
            intermediate_elements,
            output_elements,
        ]
        .into_iter()
        .enumerate()
        {
            ensure_shared_buffer(
                &self.device,
                &mut activations.outputs[index],
                checked_byte_len::<f32>(elements)?,
            )?;
        }

        let input_buffer = &activations
            .input
            .as_ref()
            .expect("input buffer is initialized")
            .buffer;
        let gate_output = &activations.outputs[0]
            .as_ref()
            .expect("gate output buffer is initialized")
            .buffer;
        let up_output = &activations.outputs[1]
            .as_ref()
            .expect("up output buffer is initialized")
            .buffer;
        let swiglu_output = &activations.outputs[2]
            .as_ref()
            .expect("SwiGLU output buffer is initialized")
            .buffer;
        let down_output = &activations.outputs[3]
            .as_ref()
            .expect("down output buffer is initialized")
            .buffer;
        copy_slice_to_buffer(input_buffer, input);

        // A single decode token should use row-wise matvecs. The tiled
        // prefill kernel has 256 threads and pays staging/barrier costs that
        // dominate this shape, while the three matvecs can still share one
        // command buffer and completion fence with SwiGLU.
        if batch_size == 1 {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            self.encode_q4_affine_matvec(encoder, input_buffer, gate_output, gate, gate_words)?;
            self.encode_q4_affine_matvec(encoder, input_buffer, up_output, up, up_words)?;
            encoder.set_compute_pipeline_state(&self.swiglu_rows);
            encoder.set_buffer(0, Some(gate_output), 0);
            encoder.set_buffer(1, Some(up_output), 0);
            encoder.set_buffer(2, Some(swiglu_output), 0);
            encoder.set_bytes(
                3,
                size_of::<u32>() as u64,
                (&swiglu_elements as *const u32).cast(),
            );
            encoder.dispatch_threads(
                MTLSize::new(u64::from(swiglu_elements), 1, 1),
                MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
            );
            self.encode_q4_affine_matvec(encoder, swiglu_output, down_output, down, down_words)?;
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            if command_buffer.status() == MTLCommandBufferStatus::Error {
                return Err(MetalRuntimeError::CommandFailed);
            }
            return Ok(unsafe {
                std::slice::from_raw_parts(down_output.contents().cast::<f32>(), output_elements)
                    .to_vec()
            });
        }

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_q4_affine_matmul(
            encoder,
            input_buffer,
            gate_output,
            gate,
            gate_words,
            batch_size,
        )?;
        self.encode_q4_affine_matmul(encoder, input_buffer, up_output, up, up_words, batch_size)?;
        encoder.set_compute_pipeline_state(&self.swiglu_rows);
        encoder.set_buffer(0, Some(gate_output), 0);
        encoder.set_buffer(1, Some(up_output), 0);
        encoder.set_buffer(2, Some(swiglu_output), 0);
        encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            (&swiglu_elements as *const u32).cast(),
        );
        encoder.dispatch_threads(
            MTLSize::new(u64::from(swiglu_elements), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        self.encode_q4_affine_matmul(
            encoder,
            swiglu_output,
            down_output,
            down,
            down_words,
            batch_size,
        )?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        Ok(unsafe {
            std::slice::from_raw_parts(down_output.contents().cast::<f32>(), output_elements)
                .to_vec()
        })
    }

    fn encode_q4_affine_dequantize_f16(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        output: &metal::Buffer,
        output_offset: u64,
        job: &MappedQ4AffineJob<'_>,
        words_per_row: u32,
    ) -> Result<(), MetalRuntimeError> {
        let output_rows = u64::try_from(job.output_rows)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("Q4 dequantize output rows"))?;
        let output_rows_u32 = u32::try_from(job.output_rows)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("Q4 dequantize output rows"))?;
        if job.aligned {
            encoder.set_compute_pipeline_state(&self.q4_affine_dequantize_f16);
            encoder.set_buffer(0, Some(job.weights), job.weight_offset);
            encoder.set_buffer(1, Some(job.scales), job.scale_offset);
            encoder.set_buffer(2, Some(job.biases), job.bias_offset);
            encoder.set_buffer(3, Some(output), output_offset);
            encoder.set_bytes(
                4,
                size_of::<u32>() as u64,
                (&words_per_row as *const u32).cast(),
            );
            encoder.set_bytes(
                5,
                size_of::<u32>() as u64,
                (&output_rows_u32 as *const u32).cast(),
            );
        } else {
            encoder.set_compute_pipeline_state(&self.q4_affine_dequantize_f16_unaligned);
            encoder.set_buffer(0, Some(job.weights), 0);
            encoder.set_buffer(1, Some(job.scales), 0);
            encoder.set_buffer(2, Some(job.biases), 0);
            encoder.set_buffer(3, Some(output), output_offset);
            encoder.set_bytes(
                4,
                size_of::<u32>() as u64,
                (&words_per_row as *const u32).cast(),
            );
            encoder.set_bytes(
                5,
                size_of::<u64>() as u64,
                (&job.weight_offset as *const u64).cast(),
            );
            encoder.set_bytes(
                6,
                size_of::<u64>() as u64,
                (&job.scale_offset as *const u64).cast(),
            );
            encoder.set_bytes(
                7,
                size_of::<u64>() as u64,
                (&job.bias_offset as *const u64).cast(),
            );
            encoder.set_bytes(
                8,
                size_of::<u32>() as u64,
                (&output_rows_u32 as *const u32).cast(),
            );
        }
        encoder.dispatch_threads(
            MTLSize::new(u64::from(words_per_row), output_rows, 1),
            MTLSize::new(Q4_MPS_THREADS_X, Q4_MPS_THREADS_Y, 1),
        );
        Ok(())
    }

    fn encode_f32_to_f16(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &metal::Buffer,
        output: &metal::Buffer,
        elements: usize,
    ) -> Result<(), MetalRuntimeError> {
        let elements = u32::try_from(elements)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("FP16 conversion elements"))?;
        encoder.set_compute_pipeline_state(&self.f32_to_f16);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, size_of::<u32>() as u64, (&elements as *const u32).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(elements), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        Ok(())
    }

    fn encode_f16_to_f32(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &metal::Buffer,
        output: &metal::Buffer,
        elements: usize,
    ) -> Result<(), MetalRuntimeError> {
        let elements = u32::try_from(elements)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("FP16 conversion elements"))?;
        encoder.set_compute_pipeline_state(&self.f16_to_f32);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, size_of::<u32>() as u64, (&elements as *const u32).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(elements), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        Ok(())
    }

    fn encode_swiglu(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate: &metal::Buffer,
        up: &metal::Buffer,
        output: &metal::Buffer,
        elements: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.swiglu_rows);
        encoder.set_buffer(0, Some(gate), 0);
        encoder.set_buffer(1, Some(up), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, size_of::<u32>() as u64, (&elements as *const u32).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(elements), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
    }

    fn encode_rms_norm(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &metal::Buffer,
        weights: &MetalF32Buffer,
        output: &metal::Buffer,
        elements: u32,
        epsilon: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rms_norm);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(&weights.buffer), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, size_of::<u32>() as u64, (&elements as *const u32).cast());
        encoder.set_bytes(4, size_of::<f32>() as u64, (&epsilon as *const f32).cast());
        encoder.set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
    }

    fn encode_add_in_place(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        destination: &metal::Buffer,
        source: &metal::Buffer,
        elements: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.add_in_place);
        encoder.set_buffer(0, Some(destination), 0);
        encoder.set_buffer(1, Some(source), 0);
        encoder.set_bytes(2, size_of::<u32>() as u64, (&elements as *const u32).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(elements), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
    }

    fn encode_swiglu_half(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate: &metal::Buffer,
        up: &metal::Buffer,
        output: &metal::Buffer,
        elements: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.swiglu_half_rows);
        encoder.set_buffer(0, Some(gate), 0);
        encoder.set_buffer(1, Some(up), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, size_of::<u32>() as u64, (&elements as *const u32).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(elements), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
    }

    fn encode_swiglu_half_split(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_and_up: &metal::Buffer,
        output: &metal::Buffer,
        intermediate_width: u32,
        elements: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.swiglu_half_split_rows);
        encoder.set_buffer(0, Some(gate_and_up), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            (&intermediate_width as *const u32).cast(),
        );
        encoder.set_bytes(3, size_of::<u32>() as u64, (&elements as *const u32).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(elements), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_gqa_prepare_query(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_with_gate: &metal::Buffer,
        q_norm: &MetalF32Buffer,
        query: &metal::Buffer,
        gate: &metal::Buffer,
        config: MetalGqaDecodeConfigU32,
        epsilon: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.gqa_prepare_query);
        encoder.set_buffer(0, Some(q_with_gate), 0);
        encoder.set_buffer(1, Some(&q_norm.buffer), 0);
        encoder.set_buffer(2, Some(query), 0);
        encoder.set_buffer(3, Some(gate), 0);
        encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&config.head_dim as *const u32).cast(),
        );
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&config.rotary_dim as *const u32).cast(),
        );
        encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            (&config.position0 as *const u32).cast(),
        );
        encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            (&config.position1 as *const u32).cast(),
        );
        encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            (&config.position2 as *const u32).cast(),
        );
        encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            (&config.section1 as *const u32).cast(),
        );
        encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            (&config.section2 as *const u32).cast(),
        );
        encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            (&config.has_mrope_sections as *const u32).cast(),
        );
        encoder.set_bytes(
            12,
            size_of::<f32>() as u64,
            (&config.rope_theta as *const f32).cast(),
        );
        encoder.set_bytes(13, size_of::<f32>() as u64, (&epsilon as *const f32).cast());
        encoder.set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(config.num_heads), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_gqa_prepare_key(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        key_input: &metal::Buffer,
        k_norm: &MetalF32Buffer,
        key_output: &metal::Buffer,
        config: MetalGqaDecodeConfigU32,
        epsilon: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.gqa_prepare_key);
        encoder.set_buffer(0, Some(key_input), 0);
        encoder.set_buffer(1, Some(&k_norm.buffer), 0);
        encoder.set_buffer(2, Some(key_output), 0);
        encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            (&config.head_dim as *const u32).cast(),
        );
        encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&config.rotary_dim as *const u32).cast(),
        );
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&config.position0 as *const u32).cast(),
        );
        encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            (&config.position1 as *const u32).cast(),
        );
        encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            (&config.position2 as *const u32).cast(),
        );
        encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            (&config.section1 as *const u32).cast(),
        );
        encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            (&config.section2 as *const u32).cast(),
        );
        encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            (&config.has_mrope_sections as *const u32).cast(),
        );
        encoder.set_bytes(
            11,
            size_of::<f32>() as u64,
            (&config.rope_theta as *const f32).cast(),
        );
        encoder.set_bytes(12, size_of::<f32>() as u64, (&epsilon as *const f32).cast());
        encoder.set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(config.kv_heads), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
    }

    fn encode_q4_affine_matvec(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &metal::Buffer,
        output: &metal::Buffer,
        job: &MappedQ4AffineJob<'_>,
        words_per_row: u32,
    ) -> Result<(), MetalRuntimeError> {
        let output_rows = u32::try_from(job.output_rows)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("output row count"))?;
        let output_groups = u64::try_from(job.output_rows.div_ceil(Q4_DECODE_OUTPUT_TILE))
            .map_err(|_| MetalRuntimeError::DimensionOverflow("decode output tile groups"))?;
        let input_elements = usize::try_from(words_per_row)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("decode input width"))?
            .checked_mul(VALUES_PER_PACKED_WORD)
            .ok_or(MetalRuntimeError::DimensionOverflow("decode input width"))?;
        let full_input_bytes = checked_byte_len::<f32>(input_elements)?;
        let tiled_input_bytes = checked_byte_len::<f32>(Q4_DECODE_TILED_INPUT_ELEMENTS)?;
        let use_mlx_fast = self.mlx_q4_decode && input_elements % 512 == 0;
        let use_shared = !use_mlx_fast
            && self.fast_q4_decode
            && input_elements <= Q4_DECODE_SHARED_MAX_INPUT_ELEMENTS
            && full_input_bytes <= self.device.max_threadgroup_memory_length();
        let use_tiled = !use_mlx_fast
            && self.fast_q4_decode
            && !use_shared
            && tiled_input_bytes <= self.device.max_threadgroup_memory_length();

        if job.aligned {
            let pipeline = if use_mlx_fast {
                &self.q4_affine_matvec_mlx_fast
            } else if use_shared {
                &self.q4_affine_matvec_shared
            } else if use_tiled {
                &self.q4_affine_matvec_tiled
            } else {
                &self.q4_affine_matvec_simd
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(job.weights), job.weight_offset);
            encoder.set_buffer(2, Some(job.scales), job.scale_offset);
            encoder.set_buffer(3, Some(job.biases), job.bias_offset);
            encoder.set_buffer(4, Some(output), 0);
            encoder.set_bytes(
                5,
                size_of::<u32>() as u64,
                (&words_per_row as *const u32).cast(),
            );
            if use_mlx_fast || use_shared || use_tiled {
                encoder.set_bytes(
                    6,
                    size_of::<u32>() as u64,
                    (&output_rows as *const u32).cast(),
                );
            }
        } else {
            let pipeline = if use_mlx_fast {
                &self.q4_affine_matvec_mlx_fast_unaligned
            } else if use_shared {
                &self.q4_affine_matvec_shared_unaligned
            } else if use_tiled {
                &self.q4_affine_matvec_tiled_unaligned
            } else {
                &self.q4_affine_matvec_simd_unaligned
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(job.weights), 0);
            encoder.set_buffer(2, Some(job.scales), 0);
            encoder.set_buffer(3, Some(job.biases), 0);
            encoder.set_buffer(4, Some(output), 0);
            encoder.set_bytes(
                5,
                size_of::<u32>() as u64,
                (&words_per_row as *const u32).cast(),
            );
            encoder.set_bytes(
                6,
                size_of::<u64>() as u64,
                (&job.weight_offset as *const u64).cast(),
            );
            encoder.set_bytes(
                7,
                size_of::<u64>() as u64,
                (&job.scale_offset as *const u64).cast(),
            );
            encoder.set_bytes(
                8,
                size_of::<u64>() as u64,
                (&job.bias_offset as *const u64).cast(),
            );
            if use_mlx_fast || use_shared || use_tiled {
                encoder.set_bytes(
                    9,
                    size_of::<u32>() as u64,
                    (&output_rows as *const u32).cast(),
                );
            }
        }

        if use_mlx_fast {
            encoder
                .dispatch_thread_groups(MTLSize::new(output_groups, 1, 1), MTLSize::new(32, 2, 1));
        } else if use_shared || use_tiled {
            let input_tile_bytes = if use_shared {
                full_input_bytes
            } else {
                tiled_input_bytes
            };
            encoder.set_threadgroup_memory_length(0, input_tile_bytes);
            encoder.dispatch_thread_groups(
                MTLSize::new(output_groups, 1, 1),
                MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
            );
        } else {
            encoder.dispatch_thread_groups(
                MTLSize::new(u64::from(output_rows), 1, 1),
                MTLSize::new(32, 1, 1),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_q4_affine_matmul(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &metal::Buffer,
        output: &metal::Buffer,
        job: &MappedQ4AffineJob<'_>,
        words_per_row: u32,
        batch_size: usize,
    ) -> Result<(), MetalRuntimeError> {
        let output_rows_u32 = u32::try_from(job.output_rows)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("output row count"))?;
        let batch_size_u32 = u32::try_from(batch_size)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("Q4 batch size"))?;
        let output_tile_groups = u64::try_from(job.output_rows.div_ceil(Q4_PREFILL_OUTPUT_TILE))
            .map_err(|_| MetalRuntimeError::DimensionOverflow("Q4 output tile groups"))?;
        let batch_tile_groups = u64::try_from(batch_size.div_ceil(Q4_PREFILL_BATCH_TILE))
            .map_err(|_| MetalRuntimeError::DimensionOverflow("Q4 batch tile groups"))?;

        // Long, aligned prompts benefit from a matrix-core path that reuses a
        // dequantized 64x8 weight tile across 64 prompt rows. Keep the
        // conservative tiled kernel for short/unaligned inputs and provide an
        // environment switch for reproducible A/B measurements.
        let use_simdgroup = self.fast_q4_prefill
            && job.aligned
            && batch_size >= Q4_SIMDGROUP_PREFILL_MIN_BATCH
            && job.output_rows >= Q4_SIMDGROUP_PREFILL_OUTPUT_TILE
            && words_per_row as usize % (Q4_SIMDGROUP_PREFILL_K_TILE / VALUES_PER_PACKED_WORD) == 0;
        let use_wide_simdgroup = self.fast_q4_prefill
            && std::env::var_os("QWEN38_DISABLE_WIDE_PREFILL").is_none()
            && job.aligned
            && batch_size >= Q4_SIMDGROUP_WIDE_PREFILL_MIN_BATCH
            && job.output_rows >= Q4_SIMDGROUP_WIDE_PREFILL_OUTPUT_TILE
            && words_per_row as usize % (Q4_SIMDGROUP_WIDE_PREFILL_K_TILE / VALUES_PER_PACKED_WORD)
                == 0
            && self
                .q4_affine_matmul_simdgroup_wide
                .max_total_threads_per_threadgroup()
                >= Q4_SIMDGROUP_WIDE_PREFILL_THREADS
            && self.device.max_threadgroup_memory_length()
                >= checked_byte_len::<u16>(Q4_SIMDGROUP_WIDE_PREFILL_INPUT_HALF)?
                    + checked_byte_len::<u16>(Q4_SIMDGROUP_WIDE_PREFILL_WEIGHT_HALF)?
                    + checked_byte_len::<f32>(Q4_SIMDGROUP_WIDE_PREFILL_OUTPUT_FLOATS)?;
        if use_wide_simdgroup {
            let simdgroup_output_tiles = u64::try_from(
                job.output_rows
                    .div_ceil(Q4_SIMDGROUP_WIDE_PREFILL_OUTPUT_TILE),
            )
            .map_err(|_| MetalRuntimeError::DimensionOverflow("wide SIMD-group output tiles"))?;
            let simdgroup_batch_tiles = u64::try_from(
                batch_size.div_ceil(Q4_SIMDGROUP_WIDE_PREFILL_BATCH_TILE),
            )
            .map_err(|_| MetalRuntimeError::DimensionOverflow("wide SIMD-group batch tiles"))?;
            encoder.set_compute_pipeline_state(&self.q4_affine_matmul_simdgroup_wide);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(job.weights), job.weight_offset);
            encoder.set_buffer(2, Some(job.scales), job.scale_offset);
            encoder.set_buffer(3, Some(job.biases), job.bias_offset);
            encoder.set_buffer(4, Some(output), 0);
            encoder.set_bytes(
                5,
                size_of::<u32>() as u64,
                (&words_per_row as *const u32).cast(),
            );
            encoder.set_bytes(
                6,
                size_of::<u32>() as u64,
                (&output_rows_u32 as *const u32).cast(),
            );
            encoder.set_bytes(
                7,
                size_of::<u32>() as u64,
                (&batch_size_u32 as *const u32).cast(),
            );
            encoder.set_threadgroup_memory_length(
                0,
                checked_byte_len::<u16>(Q4_SIMDGROUP_WIDE_PREFILL_INPUT_HALF)?,
            );
            encoder.set_threadgroup_memory_length(
                1,
                checked_byte_len::<u16>(Q4_SIMDGROUP_WIDE_PREFILL_WEIGHT_HALF)?,
            );
            encoder.set_threadgroup_memory_length(
                2,
                checked_byte_len::<f32>(Q4_SIMDGROUP_WIDE_PREFILL_OUTPUT_FLOATS)?,
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(simdgroup_output_tiles, simdgroup_batch_tiles, 1),
                MTLSize::new(Q4_SIMDGROUP_WIDE_PREFILL_THREADS, 1, 1),
            );
            return Ok(());
        }
        if use_simdgroup {
            let simdgroup_output_tiles = u64::try_from(
                job.output_rows.div_ceil(Q4_SIMDGROUP_PREFILL_OUTPUT_TILE),
            )
            .map_err(|_| MetalRuntimeError::DimensionOverflow("SIMD-group output tile groups"))?;
            let simdgroup_batch_tiles = u64::try_from(
                batch_size.div_ceil(Q4_SIMDGROUP_PREFILL_BATCH_TILE),
            )
            .map_err(|_| MetalRuntimeError::DimensionOverflow("SIMD-group batch tile groups"))?;
            encoder.set_compute_pipeline_state(&self.q4_affine_matmul_simdgroup);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(job.weights), job.weight_offset);
            encoder.set_buffer(2, Some(job.scales), job.scale_offset);
            encoder.set_buffer(3, Some(job.biases), job.bias_offset);
            encoder.set_buffer(4, Some(output), 0);
            encoder.set_bytes(
                5,
                size_of::<u32>() as u64,
                (&words_per_row as *const u32).cast(),
            );
            encoder.set_bytes(
                6,
                size_of::<u32>() as u64,
                (&output_rows_u32 as *const u32).cast(),
            );
            encoder.set_bytes(
                7,
                size_of::<u32>() as u64,
                (&batch_size_u32 as *const u32).cast(),
            );
            encoder.set_threadgroup_memory_length(
                0,
                checked_byte_len::<u16>(Q4_SIMDGROUP_PREFILL_INPUT_HALF)?,
            );
            encoder.set_threadgroup_memory_length(
                1,
                checked_byte_len::<u16>(Q4_SIMDGROUP_PREFILL_WEIGHT_HALF)?,
            );
            encoder.set_threadgroup_memory_length(
                2,
                checked_byte_len::<f32>(Q4_SIMDGROUP_PREFILL_OUTPUT_FLOATS)?,
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(simdgroup_output_tiles, simdgroup_batch_tiles, 1),
                MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
            );
            return Ok(());
        }

        let input_tile_bytes = checked_byte_len::<f32>(Q4_PREFILL_INPUT_TILE_FLOATS)?;
        let packed_tile_bytes = checked_byte_len::<u32>(Q4_PREFILL_PACKED_TILE_WORDS)?;
        let affine_tile_bytes = checked_byte_len::<f32>(Q4_PREFILL_AFFINE_TILE_FLOATS)?;
        if job.aligned {
            encoder.set_compute_pipeline_state(&self.q4_affine_matmul);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(job.weights), job.weight_offset);
            encoder.set_buffer(2, Some(job.scales), job.scale_offset);
            encoder.set_buffer(3, Some(job.biases), job.bias_offset);
            encoder.set_buffer(4, Some(output), 0);
            encoder.set_bytes(
                5,
                size_of::<u32>() as u64,
                (&words_per_row as *const u32).cast(),
            );
            encoder.set_bytes(
                6,
                size_of::<u32>() as u64,
                (&output_rows_u32 as *const u32).cast(),
            );
            encoder.set_bytes(
                7,
                size_of::<u32>() as u64,
                (&batch_size_u32 as *const u32).cast(),
            );
        } else {
            encoder.set_compute_pipeline_state(&self.q4_affine_matmul_unaligned);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(job.weights), 0);
            encoder.set_buffer(2, Some(job.scales), 0);
            encoder.set_buffer(3, Some(job.biases), 0);
            encoder.set_buffer(4, Some(output), 0);
            encoder.set_bytes(
                5,
                size_of::<u32>() as u64,
                (&words_per_row as *const u32).cast(),
            );
            encoder.set_bytes(
                6,
                size_of::<u64>() as u64,
                (&job.weight_offset as *const u64).cast(),
            );
            encoder.set_bytes(
                7,
                size_of::<u64>() as u64,
                (&job.scale_offset as *const u64).cast(),
            );
            encoder.set_bytes(
                8,
                size_of::<u64>() as u64,
                (&job.bias_offset as *const u64).cast(),
            );
            encoder.set_bytes(
                9,
                size_of::<u32>() as u64,
                (&output_rows_u32 as *const u32).cast(),
            );
            encoder.set_bytes(
                10,
                size_of::<u32>() as u64,
                (&batch_size_u32 as *const u32).cast(),
            );
        }
        encoder.set_threadgroup_memory_length(0, input_tile_bytes);
        encoder.set_threadgroup_memory_length(1, packed_tile_bytes);
        encoder.set_threadgroup_memory_length(2, affine_tile_bytes);
        encoder.dispatch_thread_groups(
            MTLSize::new(output_tile_groups, batch_tile_groups, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_q4_affine_matvec(
        &self,
        input: &metal::Buffer,
        weights: &metal::Buffer,
        weight_offset: u64,
        scales: &metal::Buffer,
        scale_offset: u64,
        biases: &metal::Buffer,
        bias_offset: u64,
        output_rows: usize,
        words_per_row: usize,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        let output_bytes = checked_byte_len::<f32>(output_rows)?;
        let output_buffer = self.device.new_buffer(
            output_bytes,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let words_per_row = u32::try_from(words_per_row)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("words per row"))?;
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let job = MappedQ4AffineJob::new(
            weights,
            weight_offset,
            scales,
            scale_offset,
            biases,
            bias_offset,
            output_rows,
            true,
        );
        self.encode_q4_affine_matvec(encoder, input, &output_buffer, &job, words_per_row)?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), output_rows).to_vec()
        };
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_q4_affine_matvec_unaligned(
        &self,
        input: &metal::Buffer,
        weights: &metal::Buffer,
        weight_offset: u64,
        scales: &metal::Buffer,
        scale_offset: u64,
        biases: &metal::Buffer,
        bias_offset: u64,
        output_rows: usize,
        words_per_row: usize,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        let output_bytes = checked_byte_len::<f32>(output_rows)?;
        let output_buffer = self.device.new_buffer(
            output_bytes,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let words_per_row = u32::try_from(words_per_row)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("words per row"))?;
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let job = MappedQ4AffineJob::new(
            weights,
            weight_offset,
            scales,
            scale_offset,
            biases,
            bias_offset,
            output_rows,
            false,
        );
        self.encode_q4_affine_matvec(encoder, input, &output_buffer, &job, words_per_row)?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), output_rows).to_vec()
        };
        Ok(output)
    }

    /// Multiply an activation matrix by an immutable row-major BF16 matrix
    /// stored in a mapped safetensors shard. Vision weights use this format,
    /// while the language model uses the Q4 affine path above.
    pub fn bf16_gemm_mapped(
        &self,
        input: &[f32],
        weights: &metal::Buffer,
        weight_offset: u64,
        input_columns: usize,
        output_columns: usize,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        if input_columns == 0 || output_columns == 0 {
            return Err(MetalRuntimeError::EmptyDimension);
        }
        if input.is_empty() || input.len() % input_columns != 0 {
            return Err(MetalRuntimeError::WrongLength {
                name: "bf16 gemm input",
                actual: input.len(),
                expected: input_columns,
            });
        }
        let input_rows = input.len() / input_columns;
        let output_elements =
            input_rows
                .checked_mul(output_columns)
                .ok_or(MetalRuntimeError::DimensionOverflow(
                    "BF16 GEMM output elements",
                ))?;
        let input_rows_u32 = u32::try_from(input_rows)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("BF16 GEMM input rows"))?;
        let input_columns_u32 = u32::try_from(input_columns)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("BF16 GEMM input columns"))?;
        let output_columns_u32 = u32::try_from(output_columns)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("BF16 GEMM output columns"))?;
        let input_buffer = self.buffer_from_slice(input)?;
        let output_buffer = self.device.new_buffer(
            checked_byte_len::<f32>(output_elements)?,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.bf16_gemm);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(weights), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<u64>() as u64,
            (&weight_offset as *const u64).cast(),
        );
        encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&input_rows_u32 as *const u32).cast(),
        );
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&input_columns_u32 as *const u32).cast(),
        );
        encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            (&output_columns_u32 as *const u32).cast(),
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                u64::try_from(output_columns.div_ceil(16))
                    .map_err(|_| MetalRuntimeError::DimensionOverflow("BF16 GEMM column groups"))?,
                u64::try_from(input_rows.div_ceil(16))
                    .map_err(|_| MetalRuntimeError::DimensionOverflow("BF16 GEMM row groups"))?,
                1,
            ),
            MTLSize::new(16, 16, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), output_elements)
                .to_vec()
        })
    }

    /// Bidirectional self-attention used by the Qwen vision tower. Scores and
    /// values remain on the GPU for the whole attention operation; the caller
    /// receives only the projected activation matrix.
    pub fn vision_attention(
        &self,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        sequence_length: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        if sequence_length == 0 || num_heads == 0 || head_dim == 0 {
            return Err(MetalRuntimeError::EmptyDimension);
        }
        if sequence_length > 1_024 {
            return Err(MetalRuntimeError::VisionSequenceTooLong {
                actual: sequence_length,
                maximum: 1_024,
            });
        }
        let head_elements =
            num_heads
                .checked_mul(head_dim)
                .ok_or(MetalRuntimeError::DimensionOverflow(
                    "vision attention head elements",
                ))?;
        let activation_elements = sequence_length.checked_mul(head_elements).ok_or(
            MetalRuntimeError::DimensionOverflow("vision attention activation elements"),
        )?;
        for (name, activation) in [
            ("vision queries", queries),
            ("vision keys", keys),
            ("vision values", values),
        ] {
            if activation.len() != activation_elements {
                return Err(MetalRuntimeError::WrongLength {
                    name,
                    actual: activation.len(),
                    expected: activation_elements,
                });
            }
        }
        let score_elements = num_heads
            .checked_mul(sequence_length)
            .and_then(|value| value.checked_mul(sequence_length))
            .ok_or(MetalRuntimeError::DimensionOverflow(
                "vision attention score elements",
            ))?;
        let sequence_length_u32 = u32::try_from(sequence_length)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("vision sequence length"))?;
        let num_heads_u32 = u32::try_from(num_heads)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("vision head count"))?;
        let head_dim_u32 = u32::try_from(head_dim)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("vision head dimension"))?;
        let query_buffer = self.buffer_from_slice(queries)?;
        let key_buffer = self.buffer_from_slice(keys)?;
        let value_buffer = self.buffer_from_slice(values)?;
        let score_buffer = self.device.new_buffer(
            checked_byte_len::<f32>(score_elements)?,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let output_buffer = self.device.new_buffer(
            checked_byte_len::<f32>(activation_elements)?,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let command_buffer = self.command_queue.new_command_buffer();

        let scores_encoder = command_buffer.new_compute_command_encoder();
        scores_encoder.set_compute_pipeline_state(&self.vision_attention_scores);
        scores_encoder.set_buffer(0, Some(&query_buffer), 0);
        scores_encoder.set_buffer(1, Some(&key_buffer), 0);
        scores_encoder.set_buffer(2, Some(&score_buffer), 0);
        scores_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            (&sequence_length_u32 as *const u32).cast(),
        );
        scores_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&num_heads_u32 as *const u32).cast(),
        );
        scores_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&head_dim_u32 as *const u32).cast(),
        );
        scores_encoder.set_threadgroup_memory_length(0, 256 * size_of::<f32>() as u64);
        scores_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(sequence_length_u32), u64::from(num_heads_u32), 1),
            MTLSize::new(256, 1, 1),
        );
        scores_encoder.end_encoding();

        let values_encoder = command_buffer.new_compute_command_encoder();
        values_encoder.set_compute_pipeline_state(&self.vision_attention_values);
        values_encoder.set_buffer(0, Some(&score_buffer), 0);
        values_encoder.set_buffer(1, Some(&value_buffer), 0);
        values_encoder.set_buffer(2, Some(&output_buffer), 0);
        values_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            (&sequence_length_u32 as *const u32).cast(),
        );
        values_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&num_heads_u32 as *const u32).cast(),
        );
        values_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&head_dim_u32 as *const u32).cast(),
        );
        values_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(sequence_length_u32), u64::from(num_heads_u32), 1),
            MTLSize::new(256, 1, 1),
        );
        values_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), activation_elements)
                .to_vec()
        })
    }

    pub fn create_deltanet_weights(
        &self,
        config: DeltaNetConfig,
        conv_weight: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        norm: &[f32],
    ) -> Result<MetalDeltaNetWeights, MetalRuntimeError> {
        validate_deltanet_config(config)?;
        let channels = config.channels()?;
        let expected_conv = channels.checked_mul(config.conv_kernel_size).ok_or(
            MetalRuntimeError::DimensionOverflow("DeltaNet convolution weights"),
        )?;
        if conv_weight.len() != expected_conv {
            return Err(MetalRuntimeError::WrongLength {
                name: "DeltaNet convolution weights",
                actual: conv_weight.len(),
                expected: expected_conv,
            });
        }
        for (name, values, expected) in [
            ("DeltaNet A_log", a_log, config.value_heads),
            ("DeltaNet dt_bias", dt_bias, config.value_heads),
            ("DeltaNet norm", norm, config.value_head_dim),
        ] {
            if values.len() != expected {
                return Err(MetalRuntimeError::WrongLength {
                    name,
                    actual: values.len(),
                    expected,
                });
            }
        }
        Ok(MetalDeltaNetWeights {
            config,
            conv_weight: self.buffer_from_slice(conv_weight)?,
            a_log: self.buffer_from_slice(a_log)?,
            dt_bias: self.buffer_from_slice(dt_bias)?,
            norm: self.buffer_from_slice(norm)?,
        })
    }

    pub fn create_deltanet_state(
        &self,
        weights: &MetalDeltaNetWeights,
    ) -> Result<MetalDeltaNetState, MetalRuntimeError> {
        let config = weights.config;
        let channels = config.channels()?;
        let history_elements = channels
            .checked_mul(config.conv_kernel_size.saturating_sub(1))
            .ok_or(MetalRuntimeError::DimensionOverflow(
                "DeltaNet convolution state",
            ))?;
        Ok(MetalDeltaNetState {
            // A one-element allocation keeps the buffer binding valid for a
            // kernel-size-one model, whose shader never reads this state.
            conv: self.zeroed_shared_buffer(checked_byte_len::<f32>(history_elements.max(1))?)?,
            recurrent: self
                .zeroed_shared_buffer(checked_byte_len::<f32>(config.recurrent_elements()?)?)?,
        })
    }

    /// Copies a completed recurrent state into request-owned Metal buffers.
    /// The state buffers use shared storage, so this is a bounded CPU memcpy
    /// after the producer command buffer has completed. Keeping this operation
    /// here prevents callers from accidentally aliasing mutable recurrent state
    /// between a prefix-cache entry and a live request.
    pub fn clone_deltanet_state(
        &self,
        source: &MetalDeltaNetState,
    ) -> Result<MetalDeltaNetState, MetalRuntimeError> {
        let conv = self.zeroed_shared_buffer(source.conv.length())?;
        let recurrent = self.zeroed_shared_buffer(source.recurrent.length())?;
        copy_buffer_bytes(&source.conv, &conv, source.conv.length())?;
        copy_buffer_bytes(&source.recurrent, &recurrent, source.recurrent.length())?;
        Ok(MetalDeltaNetState { conv, recurrent })
    }

    pub fn create_q8_kv_state(
        &self,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Q8KvState, MetalRuntimeError> {
        validate_attention_shape(kv_heads, head_dim)?;
        self.allocate_q8_kv_state(kv_heads, head_dim, 16)
    }

    pub fn reserve_q8_kv_tokens(
        &self,
        state: &mut Q8KvState,
        additional_tokens: usize,
    ) -> Result<(), MetalRuntimeError> {
        let required_tokens = state
            .sequence_length
            .checked_add(additional_tokens)
            .ok_or(MetalRuntimeError::DimensionOverflow("Q8 KV token capacity"))?;
        self.ensure_q8_kv_capacity(state, required_tokens)
    }

    /// Clones a request-local Q8 KV cache, including its allocated capacity and
    /// active sequence length. Only the active prefix is copied; unused
    /// capacity remains zeroed in the new buffers.
    pub fn clone_q8_kv_state(&self, source: &Q8KvState) -> Result<Q8KvState, MetalRuntimeError> {
        let mut cloned =
            self.allocate_q8_kv_state(source.kv_heads, source.head_dim, source.capacity_tokens)?;
        let active_elements = source
            .sequence_length
            .checked_mul(source.kv_heads)
            .and_then(|value| value.checked_mul(source.head_dim))
            .ok_or(MetalRuntimeError::DimensionOverflow(
                "active Q8 KV elements",
            ))?;
        let active_scales = source
            .sequence_length
            .checked_mul(source.kv_heads)
            .ok_or(MetalRuntimeError::DimensionOverflow("active Q8 KV scales"))?;
        copy_buffer_bytes(
            &source.keys,
            &cloned.keys,
            u64::try_from(active_elements)
                .map_err(|_| MetalRuntimeError::DimensionOverflow("active Q8 KV key bytes"))?,
        )?;
        copy_buffer_bytes(
            &source.values,
            &cloned.values,
            u64::try_from(active_elements)
                .map_err(|_| MetalRuntimeError::DimensionOverflow("active Q8 KV value bytes"))?,
        )?;
        let active_scale_bytes = checked_byte_len::<f32>(active_scales)?;
        copy_buffer_bytes(&source.key_scales, &cloned.key_scales, active_scale_bytes)?;
        copy_buffer_bytes(
            &source.value_scales,
            &cloned.value_scales,
            active_scale_bytes,
        )?;
        cloned.sequence_length = source.sequence_length;
        Ok(cloned)
    }

    fn allocate_q8_kv_state(
        &self,
        kv_heads: usize,
        head_dim: usize,
        capacity_tokens: usize,
    ) -> Result<Q8KvState, MetalRuntimeError> {
        let elements = capacity_tokens
            .checked_mul(kv_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or(MetalRuntimeError::DimensionOverflow("Q8 KV elements"))?;
        let scale_elements = capacity_tokens
            .checked_mul(kv_heads)
            .ok_or(MetalRuntimeError::DimensionOverflow("Q8 KV scale elements"))?;
        Ok(Q8KvState {
            keys: self
                .zeroed_shared_buffer(u64::try_from(elements).map_err(|_| {
                    MetalRuntimeError::DimensionOverflow("Q8 KV key byte length")
                })?)?,
            values: self
                .zeroed_shared_buffer(u64::try_from(elements).map_err(|_| {
                    MetalRuntimeError::DimensionOverflow("Q8 KV value byte length")
                })?)?,
            key_scales: self.zeroed_shared_buffer(checked_byte_len::<f32>(scale_elements)?)?,
            value_scales: self.zeroed_shared_buffer(checked_byte_len::<f32>(scale_elements)?)?,
            capacity_tokens,
            sequence_length: 0,
            kv_heads,
            head_dim,
        })
    }

    fn ensure_q8_kv_capacity(
        &self,
        state: &mut Q8KvState,
        required_tokens: usize,
    ) -> Result<(), MetalRuntimeError> {
        if required_tokens <= state.capacity_tokens {
            return Ok(());
        }
        let capacity_tokens = required_tokens
            .checked_next_power_of_two()
            .ok_or(MetalRuntimeError::DimensionOverflow("Q8 KV token capacity"))?
            .max(16);
        let replacement =
            self.allocate_q8_kv_state(state.kv_heads, state.head_dim, capacity_tokens)?;
        let active_elements = state
            .sequence_length
            .checked_mul(state.kv_heads)
            .and_then(|value| value.checked_mul(state.head_dim))
            .ok_or(MetalRuntimeError::DimensionOverflow(
                "active Q8 KV elements",
            ))?;
        let active_scales = state
            .sequence_length
            .checked_mul(state.kv_heads)
            .ok_or(MetalRuntimeError::DimensionOverflow("active Q8 KV scales"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                state.keys.contents().cast::<u8>(),
                replacement.keys.contents().cast::<u8>(),
                active_elements,
            );
            std::ptr::copy_nonoverlapping(
                state.values.contents().cast::<u8>(),
                replacement.values.contents().cast::<u8>(),
                active_elements,
            );
            std::ptr::copy_nonoverlapping(
                state.key_scales.contents().cast::<u8>(),
                replacement.key_scales.contents().cast::<u8>(),
                checked_byte_len::<f32>(active_scales)? as usize,
            );
            std::ptr::copy_nonoverlapping(
                state.value_scales.contents().cast::<u8>(),
                replacement.value_scales.contents().cast::<u8>(),
                checked_byte_len::<f32>(active_scales)? as usize,
            );
        }
        *state = replacement;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_step(
        &self,
        weights: &MetalDeltaNetWeights,
        state: &mut MetalDeltaNetState,
        qkv: &[f32],
        z: &[f32],
        b: &[f32],
        a: &[f32],
        epsilon: f32,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        let config = weights.config;
        validate_deltanet_config(config)?;
        let channels = config.channels()?;
        let value_elements = config
            .value_heads
            .checked_mul(config.value_head_dim)
            .ok_or(MetalRuntimeError::DimensionOverflow(
                "DeltaNet value elements",
            ))?;
        for (name, values, expected) in [
            ("DeltaNet qkv activation", qkv, channels),
            ("DeltaNet z activation", z, value_elements),
            ("DeltaNet b activation", b, config.value_heads),
            ("DeltaNet a activation", a, config.value_heads),
        ] {
            if values.len() != expected {
                return Err(MetalRuntimeError::WrongLength {
                    name,
                    actual: values.len(),
                    expected,
                });
            }
        }
        let dimensions = config.as_u32()?;
        let channels_u32 = u32::try_from(channels)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("DeltaNet channel count"))?;
        let mut activations = self
            .language_activations
            .lock()
            .map_err(|_| MetalRuntimeError::LanguageActivationPoolPoisoned)?;
        ensure_language_slots(
            &self.device,
            &mut activations,
            &[
                checked_byte_len::<f32>(qkv.len())?,
                checked_byte_len::<f32>(z.len())?,
                checked_byte_len::<f32>(b.len())?,
                checked_byte_len::<f32>(a.len())?,
                checked_byte_len::<f32>(channels)?,
                checked_byte_len::<f32>(value_elements)?,
            ],
        )?;
        copy_slice_to_buffer(language_slot(&activations, 0), qkv);
        copy_slice_to_buffer(language_slot(&activations, 1), z);
        copy_slice_to_buffer(language_slot(&activations, 2), b);
        copy_slice_to_buffer(language_slot(&activations, 3), a);

        let command_buffer = self.command_queue.new_command_buffer();
        let conv_encoder = command_buffer.new_compute_command_encoder();
        conv_encoder.set_compute_pipeline_state(&self.deltanet_conv);
        conv_encoder.set_buffer(0, Some(language_slot(&activations, 0)), 0);
        conv_encoder.set_buffer(1, Some(&weights.conv_weight), 0);
        conv_encoder.set_buffer(2, Some(&state.conv), 0);
        conv_encoder.set_buffer(3, Some(language_slot(&activations, 4)), 0);
        conv_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&channels_u32 as *const u32).cast(),
        );
        conv_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&dimensions.conv_kernel_size as *const u32).cast(),
        );
        conv_encoder.dispatch_threads(
            MTLSize::new(u64::from(channels_u32), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        conv_encoder.end_encoding();

        let prepare_encoder = command_buffer.new_compute_command_encoder();
        prepare_encoder.set_compute_pipeline_state(&self.deltanet_prepare);
        prepare_encoder.set_buffer(0, Some(language_slot(&activations, 4)), 0);
        prepare_encoder.set_bytes(
            1,
            size_of::<u32>() as u64,
            (&dimensions.key_heads as *const u32).cast(),
        );
        prepare_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            (&dimensions.key_head_dim as *const u32).cast(),
        );
        prepare_encoder.set_bytes(3, size_of::<f32>() as u64, (&epsilon as *const f32).cast());
        prepare_encoder
            .set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        prepare_encoder
            .set_threadgroup_memory_length(1, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        prepare_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dimensions.key_heads), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        prepare_encoder.end_encoding();

        let recurrence_encoder = command_buffer.new_compute_command_encoder();
        recurrence_encoder.set_compute_pipeline_state(&self.deltanet_recurrence);
        recurrence_encoder.set_buffer(0, Some(language_slot(&activations, 4)), 0);
        recurrence_encoder.set_buffer(1, Some(language_slot(&activations, 1)), 0);
        recurrence_encoder.set_buffer(2, Some(language_slot(&activations, 2)), 0);
        recurrence_encoder.set_buffer(3, Some(language_slot(&activations, 3)), 0);
        recurrence_encoder.set_buffer(4, Some(&weights.a_log), 0);
        recurrence_encoder.set_buffer(5, Some(&weights.dt_bias), 0);
        recurrence_encoder.set_buffer(6, Some(&state.recurrent), 0);
        recurrence_encoder.set_buffer(7, Some(language_slot(&activations, 5)), 0);
        recurrence_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            (&dimensions.key_heads as *const u32).cast(),
        );
        recurrence_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            (&dimensions.value_heads as *const u32).cast(),
        );
        recurrence_encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            (&dimensions.key_head_dim as *const u32).cast(),
        );
        recurrence_encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            (&dimensions.value_head_dim as *const u32).cast(),
        );
        recurrence_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dimensions.value_heads), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        recurrence_encoder.end_encoding();

        let gate_norm_encoder = command_buffer.new_compute_command_encoder();
        gate_norm_encoder.set_compute_pipeline_state(&self.deltanet_gate_norm);
        gate_norm_encoder.set_buffer(0, Some(language_slot(&activations, 5)), 0);
        gate_norm_encoder.set_buffer(1, Some(language_slot(&activations, 1)), 0);
        gate_norm_encoder.set_buffer(2, Some(&weights.norm), 0);
        gate_norm_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            (&dimensions.value_heads as *const u32).cast(),
        );
        gate_norm_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&dimensions.value_head_dim as *const u32).cast(),
        );
        gate_norm_encoder.set_bytes(5, size_of::<f32>() as u64, (&epsilon as *const f32).cast());
        gate_norm_encoder
            .set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        gate_norm_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dimensions.value_heads), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        gate_norm_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                language_slot(&activations, 5).contents().cast::<f32>(),
                value_elements,
            )
            .to_vec()
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_deltanet_step(
        &self,
        command_buffer: &metal::CommandBufferRef,
        weights: &MetalDeltaNetWeights,
        state: &MetalDeltaNetState,
        qkv: &metal::Buffer,
        z: &metal::Buffer,
        b: &metal::Buffer,
        a: &metal::Buffer,
        convolved: &metal::Buffer,
        output: &metal::Buffer,
        epsilon: f32,
    ) -> Result<(), MetalRuntimeError> {
        let config = weights.config;
        validate_deltanet_config(config)?;
        let channels = config.channels()?;
        let dimensions = config.as_u32()?;
        let channels_u32 = u32::try_from(channels)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("DeltaNet channel count"))?;

        let conv_encoder = command_buffer.new_compute_command_encoder();
        conv_encoder.set_compute_pipeline_state(&self.deltanet_conv);
        conv_encoder.set_buffer(0, Some(qkv), 0);
        conv_encoder.set_buffer(1, Some(&weights.conv_weight), 0);
        conv_encoder.set_buffer(2, Some(&state.conv), 0);
        conv_encoder.set_buffer(3, Some(convolved), 0);
        conv_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&channels_u32 as *const u32).cast(),
        );
        conv_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&dimensions.conv_kernel_size as *const u32).cast(),
        );
        conv_encoder.dispatch_threads(
            MTLSize::new(u64::from(channels_u32), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        conv_encoder.end_encoding();

        let prepare_encoder = command_buffer.new_compute_command_encoder();
        prepare_encoder.set_compute_pipeline_state(&self.deltanet_prepare);
        prepare_encoder.set_buffer(0, Some(convolved), 0);
        prepare_encoder.set_bytes(
            1,
            size_of::<u32>() as u64,
            (&dimensions.key_heads as *const u32).cast(),
        );
        prepare_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            (&dimensions.key_head_dim as *const u32).cast(),
        );
        prepare_encoder.set_bytes(3, size_of::<f32>() as u64, (&epsilon as *const f32).cast());
        prepare_encoder
            .set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        prepare_encoder
            .set_threadgroup_memory_length(1, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        prepare_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dimensions.key_heads), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        prepare_encoder.end_encoding();

        let recurrence_encoder = command_buffer.new_compute_command_encoder();
        recurrence_encoder.set_compute_pipeline_state(&self.deltanet_recurrence);
        recurrence_encoder.set_buffer(0, Some(convolved), 0);
        recurrence_encoder.set_buffer(1, Some(z), 0);
        recurrence_encoder.set_buffer(2, Some(b), 0);
        recurrence_encoder.set_buffer(3, Some(a), 0);
        recurrence_encoder.set_buffer(4, Some(&weights.a_log), 0);
        recurrence_encoder.set_buffer(5, Some(&weights.dt_bias), 0);
        recurrence_encoder.set_buffer(6, Some(&state.recurrent), 0);
        recurrence_encoder.set_buffer(7, Some(output), 0);
        recurrence_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            (&dimensions.key_heads as *const u32).cast(),
        );
        recurrence_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            (&dimensions.value_heads as *const u32).cast(),
        );
        recurrence_encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            (&dimensions.key_head_dim as *const u32).cast(),
        );
        recurrence_encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            (&dimensions.value_head_dim as *const u32).cast(),
        );
        recurrence_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dimensions.value_heads), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        recurrence_encoder.end_encoding();

        let gate_norm_encoder = command_buffer.new_compute_command_encoder();
        gate_norm_encoder.set_compute_pipeline_state(&self.deltanet_gate_norm);
        gate_norm_encoder.set_buffer(0, Some(output), 0);
        gate_norm_encoder.set_buffer(1, Some(z), 0);
        gate_norm_encoder.set_buffer(2, Some(&weights.norm), 0);
        gate_norm_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            (&dimensions.value_heads as *const u32).cast(),
        );
        gate_norm_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&dimensions.value_head_dim as *const u32).cast(),
        );
        gate_norm_encoder.set_bytes(5, size_of::<f32>() as u64, (&epsilon as *const f32).cast());
        gate_norm_encoder
            .set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        gate_norm_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dimensions.value_heads), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        gate_norm_encoder.end_encoding();
        Ok(())
    }

    /// Advances an entire causal DeltaNet prompt in one GPU submission. The
    /// state buffers are identical to `deltanet_step`, so decode can continue
    /// with the single-token path after prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_prefill(
        &self,
        weights: &MetalDeltaNetWeights,
        state: &mut MetalDeltaNetState,
        qkv: &[f32],
        z: &[f32],
        b: &[f32],
        a: &[f32],
        batch_size: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        let config = weights.config;
        validate_deltanet_config(config)?;
        if batch_size == 0 {
            return Err(MetalRuntimeError::EmptyDimension);
        }
        let channels = config.channels()?;
        let value_elements = config
            .value_heads
            .checked_mul(config.value_head_dim)
            .ok_or(MetalRuntimeError::DimensionOverflow(
                "DeltaNet value elements",
            ))?;
        let expected_qkv = batch_size
            .checked_mul(channels)
            .ok_or(MetalRuntimeError::DimensionOverflow("DeltaNet prefill qkv"))?;
        let expected_values =
            batch_size
                .checked_mul(value_elements)
                .ok_or(MetalRuntimeError::DimensionOverflow(
                    "DeltaNet prefill values",
                ))?;
        let expected_heads = batch_size.checked_mul(config.value_heads).ok_or(
            MetalRuntimeError::DimensionOverflow("DeltaNet prefill heads"),
        )?;
        for (name, values, expected) in [
            ("DeltaNet prefill qkv activation", qkv, expected_qkv),
            ("DeltaNet prefill z activation", z, expected_values),
            ("DeltaNet prefill b activation", b, expected_heads),
            ("DeltaNet prefill a activation", a, expected_heads),
        ] {
            if values.len() != expected {
                return Err(MetalRuntimeError::WrongLength {
                    name,
                    actual: values.len(),
                    expected,
                });
            }
        }
        let dimensions = config.as_u32()?;
        let batch_size_u32 = u32::try_from(batch_size)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("DeltaNet prefill batch"))?;
        let scratch_elements = config
            .key_head_dim
            .checked_mul(2)
            .and_then(|value| value.checked_add(THREADS_PER_THREADGROUP as usize * 3))
            .ok_or(MetalRuntimeError::DimensionOverflow(
                "DeltaNet prefill scratch",
            ))?;
        let mut activations = self
            .language_activations
            .lock()
            .map_err(|_| MetalRuntimeError::LanguageActivationPoolPoisoned)?;
        ensure_language_slots(
            &self.device,
            &mut activations,
            &[
                checked_byte_len::<f32>(qkv.len())?,
                checked_byte_len::<f32>(z.len())?,
                checked_byte_len::<f32>(b.len())?,
                checked_byte_len::<f32>(a.len())?,
                checked_byte_len::<f32>(expected_values)?,
            ],
        )?;
        copy_slice_to_buffer(language_slot(&activations, 0), qkv);
        copy_slice_to_buffer(language_slot(&activations, 1), z);
        copy_slice_to_buffer(language_slot(&activations, 2), b);
        copy_slice_to_buffer(language_slot(&activations, 3), a);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.deltanet_prefill);
        encoder.set_buffer(0, Some(language_slot(&activations, 0)), 0);
        encoder.set_buffer(1, Some(language_slot(&activations, 1)), 0);
        encoder.set_buffer(2, Some(language_slot(&activations, 2)), 0);
        encoder.set_buffer(3, Some(language_slot(&activations, 3)), 0);
        encoder.set_buffer(4, Some(&weights.conv_weight), 0);
        encoder.set_buffer(5, Some(&weights.a_log), 0);
        encoder.set_buffer(6, Some(&weights.dt_bias), 0);
        encoder.set_buffer(7, Some(&weights.norm), 0);
        encoder.set_buffer(8, Some(&state.conv), 0);
        encoder.set_buffer(9, Some(&state.recurrent), 0);
        encoder.set_buffer(10, Some(language_slot(&activations, 4)), 0);
        encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            (&batch_size_u32 as *const u32).cast(),
        );
        encoder.set_bytes(
            12,
            size_of::<u32>() as u64,
            (&dimensions.key_heads as *const u32).cast(),
        );
        encoder.set_bytes(
            13,
            size_of::<u32>() as u64,
            (&dimensions.value_heads as *const u32).cast(),
        );
        encoder.set_bytes(
            14,
            size_of::<u32>() as u64,
            (&dimensions.key_head_dim as *const u32).cast(),
        );
        encoder.set_bytes(
            15,
            size_of::<u32>() as u64,
            (&dimensions.value_head_dim as *const u32).cast(),
        );
        encoder.set_bytes(
            16,
            size_of::<u32>() as u64,
            (&dimensions.conv_kernel_size as *const u32).cast(),
        );
        encoder.set_bytes(17, size_of::<f32>() as u64, (&epsilon as *const f32).cast());
        encoder.set_threadgroup_memory_length(0, checked_byte_len::<f32>(scratch_elements)?);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dimensions.key_heads), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                language_slot(&activations, 4).contents().cast::<f32>(),
                expected_values,
            )
            .to_vec()
        })
    }

    pub fn gqa_attention_q8(
        &self,
        state: &mut Q8KvState,
        query: &[f32],
        gate: &[f32],
        key: &[f32],
        value: &[f32],
        num_heads: usize,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        validate_attention_shape(state.kv_heads, state.head_dim)?;
        if num_heads == 0 || num_heads % state.kv_heads != 0 || num_heads > u32::MAX as usize {
            return Err(MetalRuntimeError::InvalidAttentionShape);
        }
        let query_elements = num_heads
            .checked_mul(state.head_dim)
            .ok_or(MetalRuntimeError::DimensionOverflow("GQA query elements"))?;
        let key_value_elements = state
            .kv_heads
            .checked_mul(state.head_dim)
            .ok_or(MetalRuntimeError::DimensionOverflow("GQA KV elements"))?;
        for (name, values, expected) in [
            ("GQA query", query, query_elements),
            ("GQA gate", gate, query_elements),
            ("GQA key", key, key_value_elements),
            ("GQA value", value, key_value_elements),
        ] {
            if values.len() != expected {
                return Err(MetalRuntimeError::WrongLength {
                    name,
                    actual: values.len(),
                    expected,
                });
            }
        }
        let sequence_length = state
            .sequence_length
            .checked_add(1)
            .ok_or(MetalRuntimeError::DimensionOverflow("GQA sequence length"))?;
        self.ensure_q8_kv_capacity(state, sequence_length)?;
        let sequence_length_u32 = u32::try_from(sequence_length)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA sequence length"))?;
        let token_index_u32 = u32::try_from(state.sequence_length)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA token index"))?;
        let num_heads_u32 = u32::try_from(num_heads)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA head count"))?;
        let kv_heads_u32 = u32::try_from(state.kv_heads)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA KV head count"))?;
        let head_dim_u32 = u32::try_from(state.head_dim)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA head dimension"))?;
        let score_elements = num_heads
            .checked_mul(sequence_length)
            .ok_or(MetalRuntimeError::DimensionOverflow("GQA score elements"))?;
        let mut activations = self
            .language_activations
            .lock()
            .map_err(|_| MetalRuntimeError::LanguageActivationPoolPoisoned)?;
        ensure_language_slots(
            &self.device,
            &mut activations,
            &[
                checked_byte_len::<f32>(query_elements)?,
                checked_byte_len::<f32>(query_elements)?,
                checked_byte_len::<f32>(key_value_elements)?,
                checked_byte_len::<f32>(key_value_elements)?,
                checked_byte_len::<f32>(query_elements)?,
            ],
        )?;
        ensure_shared_buffer(
            &self.device,
            &mut activations.scores,
            checked_byte_len::<f32>(score_elements)?,
        )?;
        copy_slice_to_buffer(language_slot(&activations, 0), query);
        copy_slice_to_buffer(language_slot(&activations, 1), gate);
        copy_slice_to_buffer(language_slot(&activations, 2), key);
        copy_slice_to_buffer(language_slot(&activations, 3), value);
        let scores = &activations
            .scores
            .as_ref()
            .expect("GQA score buffer is initialized")
            .buffer;

        let command_buffer = self.command_queue.new_command_buffer();
        let append_encoder = command_buffer.new_compute_command_encoder();
        append_encoder.set_compute_pipeline_state(&self.q8_kv_append);
        append_encoder.set_buffer(0, Some(language_slot(&activations, 2)), 0);
        append_encoder.set_buffer(1, Some(language_slot(&activations, 3)), 0);
        append_encoder.set_buffer(2, Some(&state.keys), 0);
        append_encoder.set_buffer(3, Some(&state.values), 0);
        append_encoder.set_buffer(4, Some(&state.key_scales), 0);
        append_encoder.set_buffer(5, Some(&state.value_scales), 0);
        append_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            (&kv_heads_u32 as *const u32).cast(),
        );
        append_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            (&head_dim_u32 as *const u32).cast(),
        );
        append_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            (&token_index_u32 as *const u32).cast(),
        );
        append_encoder
            .set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        append_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(kv_heads_u32), 2, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        append_encoder.end_encoding();

        let score_encoder = command_buffer.new_compute_command_encoder();
        score_encoder.set_compute_pipeline_state(&self.gqa_q8_scores);
        score_encoder.set_buffer(0, Some(language_slot(&activations, 0)), 0);
        score_encoder.set_buffer(1, Some(&state.keys), 0);
        score_encoder.set_buffer(2, Some(&state.key_scales), 0);
        score_encoder.set_buffer(3, Some(scores), 0);
        score_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&sequence_length_u32 as *const u32).cast(),
        );
        score_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&num_heads_u32 as *const u32).cast(),
        );
        score_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            (&kv_heads_u32 as *const u32).cast(),
        );
        score_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            (&head_dim_u32 as *const u32).cast(),
        );
        score_encoder
            .set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        score_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(num_heads_u32), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        score_encoder.end_encoding();

        let value_encoder = command_buffer.new_compute_command_encoder();
        value_encoder.set_compute_pipeline_state(&self.gqa_q8_values);
        value_encoder.set_buffer(0, Some(scores), 0);
        value_encoder.set_buffer(1, Some(&state.values), 0);
        value_encoder.set_buffer(2, Some(&state.value_scales), 0);
        value_encoder.set_buffer(3, Some(language_slot(&activations, 1)), 0);
        value_encoder.set_buffer(4, Some(language_slot(&activations, 4)), 0);
        value_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&sequence_length_u32 as *const u32).cast(),
        );
        value_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            (&num_heads_u32 as *const u32).cast(),
        );
        value_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            (&kv_heads_u32 as *const u32).cast(),
        );
        value_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            (&head_dim_u32 as *const u32).cast(),
        );
        value_encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(num_heads_u32), 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        value_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        state.sequence_length = sequence_length;
        Ok(unsafe {
            std::slice::from_raw_parts(
                language_slot(&activations, 4).contents().cast::<f32>(),
                query_elements,
            )
            .to_vec()
        })
    }

    /// Appends a complete prompt span to the Q8 KV cache, then computes each
    /// causal GQA row on the GPU. The input and output are row-major by prompt
    /// position, which lets layer-major prefill retain the decode cache layout.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attention_q8_prefill(
        &self,
        state: &mut Q8KvState,
        query: &[f32],
        gate: &[f32],
        key: &[f32],
        value: &[f32],
        num_heads: usize,
        batch_size: usize,
    ) -> Result<Vec<f32>, MetalRuntimeError> {
        validate_attention_shape(state.kv_heads, state.head_dim)?;
        if batch_size == 0
            || num_heads == 0
            || num_heads % state.kv_heads != 0
            || num_heads > u32::MAX as usize
        {
            return Err(MetalRuntimeError::InvalidAttentionShape);
        }
        let query_elements_per_row = num_heads
            .checked_mul(state.head_dim)
            .ok_or(MetalRuntimeError::DimensionOverflow("GQA query elements"))?;
        let key_value_elements_per_row = state
            .kv_heads
            .checked_mul(state.head_dim)
            .ok_or(MetalRuntimeError::DimensionOverflow("GQA KV elements"))?;
        let query_elements = batch_size.checked_mul(query_elements_per_row).ok_or(
            MetalRuntimeError::DimensionOverflow("GQA prefill query elements"),
        )?;
        let key_value_elements = batch_size.checked_mul(key_value_elements_per_row).ok_or(
            MetalRuntimeError::DimensionOverflow("GQA prefill KV elements"),
        )?;
        for (name, values, expected) in [
            ("GQA prefill query", query, query_elements),
            ("GQA prefill gate", gate, query_elements),
            ("GQA prefill key", key, key_value_elements),
            ("GQA prefill value", value, key_value_elements),
        ] {
            if values.len() != expected {
                return Err(MetalRuntimeError::WrongLength {
                    name,
                    actual: values.len(),
                    expected,
                });
            }
        }

        let start_token = state.sequence_length;
        let total_length = start_token
            .checked_add(batch_size)
            .ok_or(MetalRuntimeError::DimensionOverflow("GQA sequence length"))?;
        self.ensure_q8_kv_capacity(state, total_length)?;
        let start_token_u32 = u32::try_from(start_token)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA start token"))?;
        let total_length_u32 = u32::try_from(total_length)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA sequence length"))?;
        let batch_size_u32 = u32::try_from(batch_size)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA prefill batch"))?;
        let num_heads_u32 = u32::try_from(num_heads)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA head count"))?;
        let kv_heads_u32 = u32::try_from(state.kv_heads)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA KV head count"))?;
        let head_dim_u32 = u32::try_from(state.head_dim)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA head dimension"))?;
        let append_groups = batch_size
            .checked_mul(state.kv_heads)
            .and_then(|value| value.checked_mul(2))
            .ok_or(MetalRuntimeError::DimensionOverflow(
                "GQA prefill append groups",
            ))?;
        let attention_groups =
            batch_size
                .checked_mul(num_heads)
                .ok_or(MetalRuntimeError::DimensionOverflow(
                    "GQA prefill attention groups",
                ))?;
        let append_groups_u64 = u64::try_from(append_groups)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA prefill append groups"))?;
        let attention_groups_u64 = u64::try_from(attention_groups)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("GQA prefill attention groups"))?;
        let attention_scratch_elements = THREADS_PER_THREADGROUP
            .checked_mul(2)
            .and_then(|value| value.checked_add(4))
            .ok_or(MetalRuntimeError::DimensionOverflow("GQA prefill scratch"))?;

        let mut activations = self
            .language_activations
            .lock()
            .map_err(|_| MetalRuntimeError::LanguageActivationPoolPoisoned)?;
        ensure_language_slots(
            &self.device,
            &mut activations,
            &[
                checked_byte_len::<f32>(query_elements)?,
                checked_byte_len::<f32>(query_elements)?,
                checked_byte_len::<f32>(key_value_elements)?,
                checked_byte_len::<f32>(key_value_elements)?,
                checked_byte_len::<f32>(query_elements)?,
            ],
        )?;
        copy_slice_to_buffer(language_slot(&activations, 0), query);
        copy_slice_to_buffer(language_slot(&activations, 1), gate);
        copy_slice_to_buffer(language_slot(&activations, 2), key);
        copy_slice_to_buffer(language_slot(&activations, 3), value);

        let command_buffer = self.command_queue.new_command_buffer();
        let append_encoder = command_buffer.new_compute_command_encoder();
        append_encoder.set_compute_pipeline_state(&self.q8_kv_append_prefill);
        append_encoder.set_buffer(0, Some(language_slot(&activations, 2)), 0);
        append_encoder.set_buffer(1, Some(language_slot(&activations, 3)), 0);
        append_encoder.set_buffer(2, Some(&state.keys), 0);
        append_encoder.set_buffer(3, Some(&state.values), 0);
        append_encoder.set_buffer(4, Some(&state.key_scales), 0);
        append_encoder.set_buffer(5, Some(&state.value_scales), 0);
        append_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            (&kv_heads_u32 as *const u32).cast(),
        );
        append_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            (&head_dim_u32 as *const u32).cast(),
        );
        append_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            (&start_token_u32 as *const u32).cast(),
        );
        append_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            (&batch_size_u32 as *const u32).cast(),
        );
        append_encoder
            .set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        append_encoder.dispatch_thread_groups(
            MTLSize::new(append_groups_u64, 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        append_encoder.end_encoding();

        let attention_encoder = command_buffer.new_compute_command_encoder();
        attention_encoder.set_compute_pipeline_state(&self.gqa_q8_prefill_attention);
        attention_encoder.set_buffer(0, Some(language_slot(&activations, 0)), 0);
        attention_encoder.set_buffer(1, Some(&state.keys), 0);
        attention_encoder.set_buffer(2, Some(&state.key_scales), 0);
        attention_encoder.set_buffer(3, Some(&state.values), 0);
        attention_encoder.set_buffer(4, Some(&state.value_scales), 0);
        attention_encoder.set_buffer(5, Some(language_slot(&activations, 1)), 0);
        attention_encoder.set_buffer(6, Some(language_slot(&activations, 4)), 0);
        attention_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            (&start_token_u32 as *const u32).cast(),
        );
        attention_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            (&total_length_u32 as *const u32).cast(),
        );
        attention_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            (&num_heads_u32 as *const u32).cast(),
        );
        attention_encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            (&kv_heads_u32 as *const u32).cast(),
        );
        attention_encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            (&head_dim_u32 as *const u32).cast(),
        );
        attention_encoder.set_threadgroup_memory_length(
            0,
            attention_scratch_elements
                .checked_mul(size_of::<f32>() as u64)
                .ok_or(MetalRuntimeError::DimensionOverflow("GQA prefill scratch"))?,
        );
        attention_encoder.dispatch_thread_groups(
            MTLSize::new(attention_groups_u64, 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
        attention_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(MetalRuntimeError::CommandFailed);
        }
        state.sequence_length = total_length;
        Ok(unsafe {
            std::slice::from_raw_parts(
                language_slot(&activations, 4).contents().cast::<f32>(),
                query_elements,
            )
            .to_vec()
        })
    }

    fn zeroed_shared_buffer(&self, byte_len: u64) -> Result<metal::Buffer, MetalRuntimeError> {
        if byte_len == 0 {
            return Err(MetalRuntimeError::EmptyBuffer);
        }
        let buffer = self.device.new_buffer(
            byte_len,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let byte_len = usize::try_from(byte_len)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("Metal buffer byte length"))?;
        unsafe {
            std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, byte_len);
        }
        Ok(buffer)
    }

    fn buffer_from_slice<T>(&self, values: &[T]) -> Result<metal::Buffer, MetalRuntimeError> {
        let byte_len = checked_byte_len::<T>(values.len())?;
        if byte_len == 0 {
            return Err(MetalRuntimeError::EmptyBuffer);
        }

        Ok(self.device.new_buffer_with_data(
            values.as_ptr().cast(),
            byte_len,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_mapped_q4_affine_matvec(
    input: &[f32],
    weights: &metal::Buffer,
    weight_offset: u64,
    scales: &metal::Buffer,
    scale_offset: u64,
    biases: &metal::Buffer,
    bias_offset: u64,
    output_rows: usize,
) -> Result<usize, MetalRuntimeError> {
    validate_mapped_q4_affine_shape(
        input.len(),
        weights,
        weight_offset,
        scales,
        scale_offset,
        biases,
        bias_offset,
        output_rows,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_mapped_q4_affine_shape(
    input_elements: usize,
    weights: &metal::Buffer,
    weight_offset: u64,
    scales: &metal::Buffer,
    scale_offset: u64,
    biases: &metal::Buffer,
    bias_offset: u64,
    output_rows: usize,
) -> Result<usize, MetalRuntimeError> {
    if input_elements == 0 || output_rows == 0 {
        return Err(MetalRuntimeError::EmptyDimension);
    }
    if input_elements % AFFINE_GROUP_SIZE != 0 {
        return Err(MetalRuntimeError::InputNotGrouped {
            input_elements,
            group_size: AFFINE_GROUP_SIZE,
        });
    }
    if weight_offset > weights.length()
        || scale_offset > scales.length()
        || bias_offset > biases.length()
    {
        return Err(MetalRuntimeError::InvalidBufferOffset);
    }

    let words_per_row = input_elements / VALUES_PER_PACKED_WORD;
    let required_weight_bytes = checked_byte_len::<u32>(
        output_rows
            .checked_mul(words_per_row)
            .ok_or(MetalRuntimeError::DimensionOverflow("packed weight count"))?,
    )?;
    let groups_per_row = input_elements / AFFINE_GROUP_SIZE;
    let required_affine_bytes =
        checked_byte_len::<u16>(output_rows.checked_mul(groups_per_row).ok_or(
            MetalRuntimeError::DimensionOverflow("affine parameter count"),
        )?)?;
    if required_weight_bytes > weights.length() - weight_offset
        || required_affine_bytes > scales.length() - scale_offset
        || required_affine_bytes > biases.length() - bias_offset
    {
        return Err(MetalRuntimeError::MappedTensorOutOfRange);
    }
    Ok(words_per_row)
}

fn validate_decode_q4_job(
    input_elements: usize,
    job: &MappedQ4AffineJob<'_>,
    expected_output_rows: usize,
    name: &'static str,
) -> Result<u32, MetalRuntimeError> {
    if job.output_rows != expected_output_rows {
        return Err(MetalRuntimeError::WrongLength {
            name,
            actual: job.output_rows,
            expected: expected_output_rows,
        });
    }
    let words_per_row = validate_mapped_q4_affine_shape(
        input_elements,
        job.weights,
        job.weight_offset,
        job.scales,
        job.scale_offset,
        job.biases,
        job.bias_offset,
        job.output_rows,
    )?;
    u32::try_from(words_per_row)
        .map_err(|_| MetalRuntimeError::DimensionOverflow("decode Q4 words per row"))
}

#[derive(Debug, Clone, Copy)]
struct MatvecShape {
    words_per_row: usize,
}

impl MatvecShape {
    fn validate(
        input: &[f32],
        packed_weights: &[u32],
        scales: &[u16],
        biases: &[u16],
        output_rows: usize,
    ) -> Result<Self, MetalRuntimeError> {
        if input.is_empty() || output_rows == 0 {
            return Err(MetalRuntimeError::EmptyDimension);
        }
        if input.len() % AFFINE_GROUP_SIZE != 0 {
            return Err(MetalRuntimeError::InputNotGrouped {
                input_elements: input.len(),
                group_size: AFFINE_GROUP_SIZE,
            });
        }
        let words_per_row = input.len() / VALUES_PER_PACKED_WORD;
        let group_count = input.len() / AFFINE_GROUP_SIZE;
        let expected_weights = output_rows
            .checked_mul(words_per_row)
            .ok_or(MetalRuntimeError::DimensionOverflow("packed weight count"))?;
        let expected_affine =
            output_rows
                .checked_mul(group_count)
                .ok_or(MetalRuntimeError::DimensionOverflow(
                    "affine parameter count",
                ))?;

        if packed_weights.len() != expected_weights {
            return Err(MetalRuntimeError::WrongLength {
                name: "packed_weights",
                actual: packed_weights.len(),
                expected: expected_weights,
            });
        }
        if scales.len() != expected_affine {
            return Err(MetalRuntimeError::WrongLength {
                name: "scales",
                actual: scales.len(),
                expected: expected_affine,
            });
        }
        if biases.len() != expected_affine {
            return Err(MetalRuntimeError::WrongLength {
                name: "biases",
                actual: biases.len(),
                expected: expected_affine,
            });
        }
        Ok(Self { words_per_row })
    }
}

fn checked_byte_len<T>(elements: usize) -> Result<u64, MetalRuntimeError> {
    let bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or(MetalRuntimeError::DimensionOverflow("buffer byte length"))?;
    u64::try_from(bytes).map_err(|_| MetalRuntimeError::DimensionOverflow("buffer byte length"))
}

fn ensure_shared_buffer(
    device: &Device,
    slot: &mut Option<ReusableBuffer>,
    required_bytes: u64,
) -> Result<(), MetalRuntimeError> {
    if required_bytes == 0 {
        return Err(MetalRuntimeError::EmptyBuffer);
    }
    if slot
        .as_ref()
        .is_some_and(|buffer| buffer.capacity_bytes >= required_bytes)
    {
        return Ok(());
    }
    *slot = Some(shared_reusable_buffer(device, required_bytes)?);
    Ok(())
}

fn ensure_private_buffer(
    device: &Device,
    slot: &mut Option<ReusableBuffer>,
    required_bytes: u64,
) -> Result<(), MetalRuntimeError> {
    if required_bytes == 0 {
        return Err(MetalRuntimeError::EmptyBuffer);
    }
    if slot
        .as_ref()
        .is_some_and(|buffer| buffer.capacity_bytes >= required_bytes)
    {
        return Ok(());
    }
    let capacity_bytes = required_bytes
        .checked_next_power_of_two()
        .unwrap_or(required_bytes);
    let options = if std::env::var_os("QWEN38_MPS_SHARED_SCRATCH").is_some() {
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache
    } else {
        MTLResourceOptions::StorageModePrivate
    };
    *slot = Some(ReusableBuffer {
        buffer: device.new_buffer(capacity_bytes, options),
        capacity_bytes,
    });
    Ok(())
}

fn shared_reusable_buffer(
    device: &Device,
    required_bytes: u64,
) -> Result<ReusableBuffer, MetalRuntimeError> {
    if required_bytes == 0 {
        return Err(MetalRuntimeError::EmptyBuffer);
    }
    let capacity_bytes = required_bytes
        .checked_next_power_of_two()
        .unwrap_or(required_bytes);
    Ok(ReusableBuffer {
        buffer: device.new_buffer(
            capacity_bytes,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        ),
        capacity_bytes,
    })
}

fn copy_slice_to_buffer<T>(buffer: &metal::Buffer, values: &[T]) {
    let byte_len = values.len().saturating_mul(size_of::<T>());
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().cast::<u8>(),
            byte_len,
        );
    }
}

fn copy_buffer_bytes(
    source: &metal::Buffer,
    destination: &metal::Buffer,
    byte_len: u64,
) -> Result<(), MetalRuntimeError> {
    if byte_len > source.length() || byte_len > destination.length() {
        return Err(MetalRuntimeError::WrongLength {
            name: "Metal state buffer copy",
            actual: usize::try_from(byte_len).unwrap_or(usize::MAX),
            expected: usize::try_from(source.length().min(destination.length()))
                .unwrap_or(usize::MAX),
        });
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            source.contents().cast::<u8>(),
            destination.contents().cast::<u8>(),
            usize::try_from(byte_len)
                .map_err(|_| MetalRuntimeError::DimensionOverflow("Metal state copy bytes"))?,
        );
    }
    Ok(())
}

fn ensure_language_slots(
    device: &Device,
    activations: &mut LanguageActivationPool,
    required_bytes: &[u64],
) -> Result<(), MetalRuntimeError> {
    if activations.slots.len() < required_bytes.len() {
        activations.slots.resize_with(required_bytes.len(), || None);
    }
    for (slot, bytes) in activations.slots.iter_mut().zip(required_bytes) {
        ensure_shared_buffer(device, slot, *bytes)?;
    }
    Ok(())
}

fn language_slot(activations: &LanguageActivationPool, index: usize) -> &metal::Buffer {
    &activations.slots[index]
        .as_ref()
        .expect("language activation slot is initialized")
        .buffer
}

fn validate_deltanet_config(config: DeltaNetConfig) -> Result<(), MetalRuntimeError> {
    if config.key_heads == 0
        || config.value_heads == 0
        || config.key_head_dim == 0
        || config.value_head_dim == 0
        || config.conv_kernel_size == 0
        || config.value_heads % config.key_heads != 0
        || config.key_head_dim > THREADS_PER_THREADGROUP as usize
        || config.value_head_dim > THREADS_PER_THREADGROUP as usize
    {
        return Err(MetalRuntimeError::InvalidDeltaNetConfig);
    }
    config.channels()?;
    config.recurrent_elements()?;
    config.as_u32()?;
    Ok(())
}

fn validate_attention_shape(kv_heads: usize, head_dim: usize) -> Result<(), MetalRuntimeError> {
    if kv_heads == 0 || head_dim == 0 || head_dim > THREADS_PER_THREADGROUP as usize {
        return Err(MetalRuntimeError::InvalidAttentionShape);
    }
    u32::try_from(kv_heads).map_err(|_| MetalRuntimeError::InvalidAttentionShape)?;
    u32::try_from(head_dim).map_err(|_| MetalRuntimeError::InvalidAttentionShape)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalRuntimeError {
    NoDevice,
    Library(String),
    Function(String),
    Pipeline(String),
    UnsupportedThreadgroupLimit {
        available: u64,
        required: u64,
    },
    EmptyBuffer,
    ActivationPoolPoisoned,
    LanguageActivationPoolPoisoned,
    EmptyDimension,
    InvalidDeltaNetConfig,
    InvalidAttentionShape,
    InputNotGrouped {
        input_elements: usize,
        group_size: usize,
    },
    WrongLength {
        name: &'static str,
        actual: usize,
        expected: usize,
    },
    MissingMappedShard(usize),
    InvalidBufferOffset,
    MappedTensorOutOfRange,
    DimensionOverflow(&'static str),
    VisionSequenceTooLong {
        actual: usize,
        maximum: usize,
    },
    Mps(String),
    CommandFailed,
}

impl fmt::Display for MetalRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice => write!(formatter, "no Metal device is available"),
            Self::Library(error) => write!(formatter, "cannot load embedded Metal library: {error}"),
            Self::Function(error) => write!(formatter, "cannot load Metal function: {error}"),
            Self::Pipeline(error) => write!(formatter, "cannot create Metal pipeline: {error}"),
            Self::UnsupportedThreadgroupLimit { available, required } => write!(
                formatter,
                "Metal device supports {available} threads per threadgroup but the Q4 kernel requires {required}"
            ),
            Self::EmptyBuffer => write!(formatter, "Metal buffers cannot be empty"),
            Self::ActivationPoolPoisoned => {
                write!(formatter, "the reusable Metal activation pool is unavailable")
            }
            Self::LanguageActivationPoolPoisoned => {
                write!(formatter, "the reusable language activation pool is unavailable")
            }
            Self::EmptyDimension => write!(formatter, "matrix dimensions must be greater than zero"),
            Self::InvalidDeltaNetConfig => write!(
                formatter,
                "DeltaNet dimensions must be non-zero, GPU-threadgroup compatible, and evenly grouped"
            ),
            Self::InvalidAttentionShape => write!(
                formatter,
                "attention dimensions must be non-zero and fit one GPU threadgroup"
            ),
            Self::InputNotGrouped {
                input_elements,
                group_size,
            } => write!(
                formatter,
                "input dimension {input_elements} is not divisible by affine group size {group_size}"
            ),
            Self::WrongLength {
                name,
                actual,
                expected,
            } => write!(formatter, "{name} has {actual} elements, expected {expected}"),
            Self::MissingMappedShard(index) => {
                write!(formatter, "mapped safetensors shard {index} is absent")
            }
            Self::InvalidBufferOffset => write!(formatter, "mapped tensor offset is outside its shard"),
            Self::MappedTensorOutOfRange => {
                write!(formatter, "mapped tensor range exceeds its safetensors shard")
            }
            Self::DimensionOverflow(name) => write!(formatter, "{name} exceeds runtime limits"),
            Self::VisionSequenceTooLong { actual, maximum } => write!(
                formatter,
                "vision sequence has {actual} patches, exceeding the {maximum}-patch GPU attention limit"
            ),
            Self::Mps(error) => write!(formatter, "MPS matrix operation failed: {error}"),
            Self::CommandFailed => write!(formatter, "Metal command buffer failed"),
        }
    }
}

impl Error for MetalRuntimeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_q4_affine_matvec_matches_cpu_reference() {
        let input: Vec<f32> = (0..128)
            .map(|index| ((index % 17) as f32 - 8.0) * 0.125)
            .collect();
        let output_rows = 5;
        let quantized: Vec<u8> = (0..output_rows * input.len())
            .map(|index| ((index * 7 + 3) % 16) as u8)
            .collect();
        let packed_weights = pack_q4(&quantized);
        let scale_values = [0.125, -0.0625, 0.25, 0.03125, -0.125, 0.0625];
        let bias_values = [0.03125, -0.015625, 0.0625, 0.125, -0.03125, 0.0];
        let scales: Vec<u16> = (0..output_rows * 2)
            .map(|index| f32_to_bf16(scale_values[index % scale_values.len()]))
            .collect();
        let biases: Vec<u16> = (0..output_rows * 2)
            .map(|index| f32_to_bf16(bias_values[index % bias_values.len()]))
            .collect();

        let runtime = MetalRuntime::new().unwrap();
        let actual = runtime
            .q4_affine_matvec(&input, &packed_weights, &scales, &biases, output_rows)
            .unwrap();
        let expected = cpu_q4_affine_matvec(&input, &quantized, &scales, &biases, output_rows);

        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.001,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn q4_affine_matvec_matches_cpu_reference() {
        assert_q4_affine_matvec_matches_cpu_reference();
    }

    #[test]
    fn gqa_gpu_prepare_matches_rms_norm_and_mrope() {
        let config = MetalGqaDecodeConfig {
            num_heads: 2,
            kv_heads: 1,
            head_dim: 4,
            rotary_dim: 4,
            position: [7, 11, 13],
            section1: 1,
            section2: 1,
            has_mrope_sections: true,
            rope_theta: 10_000.0,
        };
        let epsilon = 1e-6;
        let q_with_gate = vec![
            0.25, -0.5, 0.75, -1.0, 0.1, 0.2, 0.3, 0.4, -0.3, 0.6, -0.9, 1.2, -0.2, -0.1, 0.0, 0.1,
        ];
        let key_input = vec![0.5, -0.25, 0.75, -1.0];
        let q_norm = vec![0.9, 1.0, 1.1, 1.2];
        let k_norm = vec![1.2, 1.1, 1.0, 0.9];
        let runtime = MetalRuntime::new().unwrap();
        let q_with_gate_buffer = runtime.buffer_from_slice(&q_with_gate).unwrap();
        let key_input_buffer = runtime.buffer_from_slice(&key_input).unwrap();
        let q_norm_buffer = runtime.create_f32_buffer(&q_norm).unwrap();
        let k_norm_buffer = runtime.create_f32_buffer(&k_norm).unwrap();
        let query_buffer = runtime
            .zeroed_shared_buffer(checked_byte_len::<f32>(8).unwrap())
            .unwrap();
        let gate_buffer = runtime
            .zeroed_shared_buffer(checked_byte_len::<f32>(8).unwrap())
            .unwrap();
        let key_output_buffer = runtime
            .zeroed_shared_buffer(checked_byte_len::<f32>(4).unwrap())
            .unwrap();
        let command_buffer = runtime.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let config_u32 = config.as_u32().unwrap();
        runtime.encode_gqa_prepare_query(
            encoder,
            &q_with_gate_buffer,
            &q_norm_buffer,
            &query_buffer,
            &gate_buffer,
            config_u32,
            epsilon,
        );
        runtime.encode_gqa_prepare_key(
            encoder,
            &key_input_buffer,
            &k_norm_buffer,
            &key_output_buffer,
            config_u32,
            epsilon,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_ne!(command_buffer.status(), MTLCommandBufferStatus::Error);

        let normalize = |values: &[f32], weights: &[f32]| {
            let mean = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
            let inverse = (mean + epsilon).sqrt().recip();
            values
                .iter()
                .zip(weights)
                .map(|(value, weight)| value * inverse * weight)
                .collect::<Vec<_>>()
        };
        let rotate = |values: &mut [f32]| {
            let half = config.rotary_dim / 2;
            for index in 0..half {
                let axis = match index % 3 {
                    1 if index / 3 < config.section1 as usize => config.position[1],
                    2 if index / 3 < config.section2 as usize => config.position[2],
                    _ => config.position[0],
                };
                let exponent = (2 * index) as f32 / config.rotary_dim as f32;
                let angle = axis as f32 / config.rope_theta.powf(exponent);
                let (sine, cosine) = angle.sin_cos();
                let left = values[index];
                let right = values[index + half];
                values[index] = left * cosine - right * sine;
                values[index + half] = right * cosine + left * sine;
            }
        };
        let mut expected_query = Vec::new();
        let mut expected_gate = Vec::new();
        for head in 0..config.num_heads {
            let offset = head * config.head_dim * 2;
            let mut values = normalize(&q_with_gate[offset..offset + config.head_dim], &q_norm);
            rotate(&mut values);
            expected_query.extend(values);
            expected_gate.extend_from_slice(
                &q_with_gate[offset + config.head_dim..offset + config.head_dim * 2],
            );
        }
        let mut expected_key = normalize(&key_input, &k_norm);
        rotate(&mut expected_key);
        let actual_query = unsafe {
            std::slice::from_raw_parts(query_buffer.contents().cast::<f32>(), 8).to_vec()
        };
        let actual_gate =
            unsafe { std::slice::from_raw_parts(gate_buffer.contents().cast::<f32>(), 8).to_vec() };
        let actual_key = unsafe {
            std::slice::from_raw_parts(key_output_buffer.contents().cast::<f32>(), 4).to_vec()
        };
        for (actual, expected) in actual_query.into_iter().zip(expected_query) {
            assert!(
                (actual - expected).abs() < 0.001,
                "actual {actual}, expected {expected}"
            );
        }
        for (actual, expected) in actual_gate.into_iter().zip(expected_gate) {
            assert!(
                (actual - expected).abs() < 0.001,
                "actual {actual}, expected {expected}"
            );
        }
        for (actual, expected) in actual_key.into_iter().zip(expected_key) {
            assert!(
                (actual - expected).abs() < 0.001,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn gpu_decode_full_layer_matches_the_segmented_path() {
        const HIDDEN: usize = 64;
        let epsilon = 1e-6;
        let runtime = MetalRuntime::new().unwrap();
        let make_q4_buffers = |output_rows: usize, seed: usize| {
            let quantized: Vec<u8> = (0..output_rows * HIDDEN)
                .map(|index| ((index * 11 + seed) % 16) as u8)
                .collect();
            let scales = (0..output_rows)
                .map(|index| f32_to_bf16(0.01171875 + (index % 5) as f32 * 0.00390625))
                .collect::<Vec<_>>();
            let biases = (0..output_rows)
                .map(|index| f32_to_bf16(-0.015625 + (index % 3) as f32 * 0.00390625))
                .collect::<Vec<_>>();
            (
                runtime.buffer_from_slice(&pack_q4(&quantized)).unwrap(),
                runtime.buffer_from_slice(&scales).unwrap(),
                runtime.buffer_from_slice(&biases).unwrap(),
            )
        };
        let gqa = MetalGqaDecodeConfig {
            num_heads: 1,
            kv_heads: 1,
            head_dim: HIDDEN,
            rotary_dim: HIDDEN,
            position: [3, 5, 7],
            section1: 11,
            section2: 10,
            has_mrope_sections: true,
            rope_theta: 10_000.0,
        };
        let (q_weights, q_scales, q_biases) = make_q4_buffers(HIDDEN * 2, 1);
        let (k_weights, k_scales, k_biases) = make_q4_buffers(HIDDEN, 2);
        let (v_weights, v_scales, v_biases) = make_q4_buffers(HIDDEN, 3);
        let (o_weights, o_scales, o_biases) = make_q4_buffers(HIDDEN, 4);
        let (gate_weights, gate_scales, gate_biases) = make_q4_buffers(HIDDEN, 5);
        let (up_weights, up_scales, up_biases) = make_q4_buffers(HIDDEN, 6);
        let (down_weights, down_scales, down_biases) = make_q4_buffers(HIDDEN, 7);
        let q_job =
            MappedQ4AffineJob::new(&q_weights, 0, &q_scales, 0, &q_biases, 0, HIDDEN * 2, true);
        let k_job = MappedQ4AffineJob::new(&k_weights, 0, &k_scales, 0, &k_biases, 0, HIDDEN, true);
        let v_job = MappedQ4AffineJob::new(&v_weights, 0, &v_scales, 0, &v_biases, 0, HIDDEN, true);
        let o_job = MappedQ4AffineJob::new(&o_weights, 0, &o_scales, 0, &o_biases, 0, HIDDEN, true);
        let gate_job = MappedQ4AffineJob::new(
            &gate_weights,
            0,
            &gate_scales,
            0,
            &gate_biases,
            0,
            HIDDEN,
            true,
        );
        let up_job =
            MappedQ4AffineJob::new(&up_weights, 0, &up_scales, 0, &up_biases, 0, HIDDEN, true);
        let down_job = MappedQ4AffineJob::new(
            &down_weights,
            0,
            &down_scales,
            0,
            &down_biases,
            0,
            HIDDEN,
            true,
        );
        let input_norm: Vec<f32> = (0..HIDDEN)
            .map(|index| 0.9 + (index % 7) as f32 * 0.015625)
            .collect();
        let post_norm: Vec<f32> = (0..HIDDEN)
            .map(|index| 0.95 + (index % 5) as f32 * 0.015625)
            .collect();
        let q_norm: Vec<f32> = (0..HIDDEN)
            .map(|index| 0.85 + (index % 3) as f32 * 0.03125)
            .collect();
        let k_norm: Vec<f32> = (0..HIDDEN)
            .map(|index| 1.0 + (index % 4) as f32 * 0.015625)
            .collect();
        let input_norm_gpu = runtime.create_f32_buffer(&input_norm).unwrap();
        let post_norm_gpu = runtime.create_f32_buffer(&post_norm).unwrap();
        let q_norm_gpu = runtime.create_f32_buffer(&q_norm).unwrap();
        let k_norm_gpu = runtime.create_f32_buffer(&k_norm).unwrap();
        let layer = MetalDecodeFullLayer::new(
            &input_norm_gpu,
            &post_norm_gpu,
            q_job,
            k_job,
            v_job,
            o_job,
            &q_norm_gpu,
            &k_norm_gpu,
            gqa,
            gate_job,
            up_job,
            down_job,
        );
        let normalize = |values: &[f32], weights: &[f32]| {
            let mean = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
            let inverse = (mean + epsilon).sqrt().recip();
            values
                .iter()
                .zip(weights)
                .map(|(value, weight)| value * inverse * weight)
                .collect::<Vec<_>>()
        };
        let prepare = |values: &[f32], weights: &[f32], heads: usize| {
            let mut result = Vec::with_capacity(values.len());
            for head in 0..heads {
                let offset = head * HIDDEN;
                let mut normalized = normalize(&values[offset..offset + HIDDEN], weights);
                for index in 0..HIDDEN / 2 {
                    let axis = match index % 3 {
                        1 if index / 3 < gqa.section1 as usize => gqa.position[1],
                        2 if index / 3 < gqa.section2 as usize => gqa.position[2],
                        _ => gqa.position[0],
                    };
                    let exponent = (2 * index) as f32 / HIDDEN as f32;
                    let angle = axis as f32 / gqa.rope_theta.powf(exponent);
                    let (sine, cosine) = angle.sin_cos();
                    let left = normalized[index];
                    let right = normalized[index + HIDDEN / 2];
                    normalized[index] = left * cosine - right * sine;
                    normalized[index + HIDDEN / 2] = right * cosine + left * sine;
                }
                result.extend(normalized);
            }
            result
        };
        let mut segmented_kv = runtime.create_q8_kv_state(1, HIDDEN).unwrap();
        let mut gpu_kv = runtime.create_q8_kv_state(1, HIDDEN).unwrap();
        let mut decode = runtime.create_decode_state(HIDDEN).unwrap();
        let inputs = [
            (0..HIDDEN)
                .map(|index| ((index % 17) as f32 - 8.0) * 0.0625)
                .collect::<Vec<_>>(),
            (0..HIDDEN)
                .map(|index| ((index % 19) as f32 - 9.0) * 0.046875)
                .collect::<Vec<_>>(),
        ];

        for input in inputs {
            let normalized = normalize(&input, &input_norm);
            let mut projections = runtime
                .q4_affine_matvec_mapped_batch(&normalized, &[q_job, k_job, v_job])
                .unwrap();
            let q_with_gate = projections.remove(0);
            let raw_key = projections.remove(0);
            let value = projections.remove(0);
            let query = prepare(&q_with_gate[..HIDDEN], &q_norm, 1);
            let gate = q_with_gate[HIDDEN..].to_vec();
            let key = prepare(&raw_key, &k_norm, 1);
            let attention = runtime
                .gqa_attention_q8(&mut segmented_kv, &query, &gate, &key, &value, 1)
                .unwrap();
            let mixed = runtime
                .q4_affine_matvec_mapped_batch(&attention, &[o_job])
                .unwrap()
                .remove(0);
            let mut expected: Vec<f32> = input
                .iter()
                .zip(mixed)
                .map(|(hidden, mixed)| hidden + mixed)
                .collect();
            let post = normalize(&expected, &post_norm);
            let mlp = runtime
                .q4_affine_mlp_mapped_batch(&post, 1, &gate_job, &up_job, &down_job)
                .unwrap();
            for (hidden, mlp) in expected.iter_mut().zip(mlp) {
                *hidden += mlp;
            }

            runtime.write_decode_hidden(&mut decode, &input).unwrap();
            {
                let mut graph = [MetalDecodeLayer::Full(layer, &mut gpu_kv)];
                runtime
                    .decode_layers(&mut decode, &mut graph, epsilon)
                    .unwrap();
            }
            let actual = runtime.read_decode_hidden(&decode).unwrap();
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 0.004,
                    "actual {actual}, expected {expected}"
                );
            }
        }
        assert_eq!(segmented_kv.sequence_length, gpu_kv.sequence_length);
    }

    #[test]
    fn mapped_q4_batch_reuses_one_input_without_changing_results() {
        let input: Vec<f32> = (0..128)
            .map(|index| ((index % 13) as f32 - 6.0) * 0.125)
            .collect();
        let first_rows = 2;
        let second_rows = 4;
        let first_quantized: Vec<u8> = (0..first_rows * input.len())
            .map(|index| ((index * 5 + 1) % 16) as u8)
            .collect();
        let second_quantized: Vec<u8> = (0..second_rows * input.len())
            .map(|index| ((index * 3 + 7) % 16) as u8)
            .collect();
        let first_scales = vec![f32_to_bf16(0.125); first_rows * 2];
        let first_biases = vec![f32_to_bf16(-0.03125); first_rows * 2];
        let second_scales = vec![f32_to_bf16(0.0625); second_rows * 2];
        let second_biases = vec![f32_to_bf16(0.015625); second_rows * 2];
        let runtime = MetalRuntime::new().unwrap();
        let first_weights = runtime
            .buffer_from_slice(&pack_q4(&first_quantized))
            .unwrap();
        let first_scales_buffer = runtime.buffer_from_slice(&first_scales).unwrap();
        let first_biases_buffer = runtime.buffer_from_slice(&first_biases).unwrap();
        let second_weights = runtime
            .buffer_from_slice(&pack_q4(&second_quantized))
            .unwrap();
        let second_scales_buffer = runtime.buffer_from_slice(&second_scales).unwrap();
        let second_biases_buffer = runtime.buffer_from_slice(&second_biases).unwrap();
        let actual = runtime
            .q4_affine_matvec_mapped_batch(
                &input,
                &[
                    MappedQ4AffineJob::new(
                        &first_weights,
                        0,
                        &first_scales_buffer,
                        0,
                        &first_biases_buffer,
                        0,
                        first_rows,
                        true,
                    ),
                    MappedQ4AffineJob::new(
                        &second_weights,
                        0,
                        &second_scales_buffer,
                        0,
                        &second_biases_buffer,
                        0,
                        second_rows,
                        true,
                    ),
                ],
            )
            .unwrap();
        let expected = [
            cpu_q4_affine_matvec(
                &input,
                &first_quantized,
                &first_scales,
                &first_biases,
                first_rows,
            ),
            cpu_q4_affine_matvec(
                &input,
                &second_quantized,
                &second_scales,
                &second_biases,
                second_rows,
            ),
        ];
        for (actual, expected) in actual.into_iter().zip(expected) {
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 0.001,
                    "actual {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn mapped_q4_matmul_batch_matches_each_prompt_row() {
        let batch_size = 3;
        let input_width = 128;
        let output_rows = 3;
        let input: Vec<f32> = (0..batch_size * input_width)
            .map(|index| ((index % 23) as f32 - 11.0) * 0.0625)
            .collect();
        let quantized: Vec<u8> = (0..output_rows * input_width)
            .map(|index| ((index * 7 + 5) % 16) as u8)
            .collect();
        let scales = vec![f32_to_bf16(0.125); output_rows * 2];
        let biases = vec![f32_to_bf16(-0.03125); output_rows * 2];
        let runtime = MetalRuntime::new().unwrap();
        let weight_buffer = runtime.buffer_from_slice(&pack_q4(&quantized)).unwrap();
        let scale_buffer = runtime.buffer_from_slice(&scales).unwrap();
        let bias_buffer = runtime.buffer_from_slice(&biases).unwrap();
        let actual = runtime
            .q4_affine_matmul_mapped_batch(
                &input,
                batch_size,
                &[MappedQ4AffineJob::new(
                    &weight_buffer,
                    0,
                    &scale_buffer,
                    0,
                    &bias_buffer,
                    0,
                    output_rows,
                    true,
                )],
            )
            .unwrap()
            .remove(0);
        let expected: Vec<f32> = input
            .chunks_exact(input_width)
            .flat_map(|row| cpu_q4_affine_matvec(row, &quantized, &scales, &biases, output_rows))
            .collect();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.001,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn mapped_q4_matmul_unaligned_batch_matches_each_prompt_row() {
        let batch_size = Q4_MPS_PREFILL_MIN_BATCH;
        let input_width = 128;
        let output_rows = 3;
        let input: Vec<f32> = (0..batch_size * input_width)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.046875)
            .collect();
        let quantized: Vec<u8> = (0..output_rows * input_width)
            .map(|index| ((index * 9 + 2) % 16) as u8)
            .collect();
        let scales = vec![f32_to_bf16(0.09375); output_rows * 2];
        let biases = vec![f32_to_bf16(-0.0234375); output_rows * 2];
        let packed = pack_q4(&quantized);
        let mut weight_bytes = vec![0_u8];
        let mut scale_bytes = vec![0_u8];
        let mut bias_bytes = vec![0_u8];
        for value in packed {
            weight_bytes.extend(value.to_le_bytes());
        }
        for value in &scales {
            scale_bytes.extend(value.to_le_bytes());
        }
        for value in &biases {
            bias_bytes.extend(value.to_le_bytes());
        }
        let runtime = MetalRuntime::new().unwrap();
        let weight_buffer = runtime.buffer_from_slice(&weight_bytes).unwrap();
        let scale_buffer = runtime.buffer_from_slice(&scale_bytes).unwrap();
        let bias_buffer = runtime.buffer_from_slice(&bias_bytes).unwrap();
        let actual = runtime
            .q4_affine_matmul_mapped_batch(
                &input,
                batch_size,
                &[MappedQ4AffineJob::new(
                    &weight_buffer,
                    1,
                    &scale_buffer,
                    1,
                    &bias_buffer,
                    1,
                    output_rows,
                    false,
                )],
            )
            .unwrap()
            .remove(0);
        let expected: Vec<f32> = input
            .chunks_exact(input_width)
            .flat_map(|row| cpu_q4_affine_matvec(row, &quantized, &scales, &biases, output_rows))
            .collect();
        let tolerance = if runtime.mps_q4_prefill { 0.08 } else { 0.001 };
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < tolerance,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn simdgroup_q4_matmul_batch_matches_cpu_reference() {
        // The dimensions and batch size cross the fast-path thresholds while
        // keeping this test small enough to run quickly on the developer GPU.
        let batch_size = 64;
        let input_width = 128;
        let output_rows = 11;
        let input: Vec<f32> = (0..batch_size * input_width)
            .map(|index| ((index % 37) as f32 - 18.0) * 0.03125)
            .collect();
        let quantized: Vec<u8> = (0..output_rows * input_width)
            .map(|index| ((index * 5 + 9) % 16) as u8)
            .collect();
        let scales = (0..output_rows * (input_width / AFFINE_GROUP_SIZE))
            .map(|index| f32_to_bf16(0.0625 + (index % 3) as f32 * 0.015625))
            .collect::<Vec<_>>();
        let biases = (0..output_rows * (input_width / AFFINE_GROUP_SIZE))
            .map(|index| f32_to_bf16(-0.015625 + (index % 2) as f32 * 0.0078125))
            .collect::<Vec<_>>();
        let runtime = MetalRuntime::new().unwrap();
        let weight_buffer = runtime.buffer_from_slice(&pack_q4(&quantized)).unwrap();
        let scale_buffer = runtime.buffer_from_slice(&scales).unwrap();
        let bias_buffer = runtime.buffer_from_slice(&biases).unwrap();
        let actual = runtime
            .q4_affine_matmul_mapped_batch(
                &input,
                batch_size,
                &[MappedQ4AffineJob::new(
                    &weight_buffer,
                    0,
                    &scale_buffer,
                    0,
                    &bias_buffer,
                    0,
                    output_rows,
                    true,
                )],
            )
            .unwrap()
            .remove(0);
        let expected: Vec<f32> = input
            .chunks_exact(input_width)
            .flat_map(|row| cpu_q4_affine_matvec(row, &quantized, &scales, &biases, output_rows))
            .collect();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            // The fast path stages dequantized weights and activations as
            // half, so permit the corresponding bounded round-off.
            assert!(
                (actual - expected).abs() < 0.08,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn wide_simdgroup_q4_matmul_batch_matches_cpu_reference() {
        let batch_size = 3072;
        let input_width = 128;
        let output_rows = 19;
        let input: Vec<f32> = (0..batch_size * input_width)
            .map(|index| ((index % 41) as f32 - 20.0) * 0.0234375)
            .collect();
        let quantized: Vec<u8> = (0..output_rows * input_width)
            .map(|index| ((index * 13 + 2) % 16) as u8)
            .collect();
        let scales = (0..output_rows * (input_width / AFFINE_GROUP_SIZE))
            .map(|index| f32_to_bf16(0.046875 + (index % 4) as f32 * 0.01171875))
            .collect::<Vec<_>>();
        let biases = (0..output_rows * (input_width / AFFINE_GROUP_SIZE))
            .map(|index| f32_to_bf16(-0.01171875 + (index % 3) as f32 * 0.00390625))
            .collect::<Vec<_>>();
        let runtime = MetalRuntime::new().unwrap();
        let weight_buffer = runtime.buffer_from_slice(&pack_q4(&quantized)).unwrap();
        let scale_buffer = runtime.buffer_from_slice(&scales).unwrap();
        let bias_buffer = runtime.buffer_from_slice(&biases).unwrap();
        let actual = runtime
            .q4_affine_matmul_mapped_batch(
                &input,
                batch_size,
                &[MappedQ4AffineJob::new(
                    &weight_buffer,
                    0,
                    &scale_buffer,
                    0,
                    &bias_buffer,
                    0,
                    output_rows,
                    true,
                )],
            )
            .unwrap()
            .remove(0);
        let expected: Vec<f32> = input
            .chunks_exact(input_width)
            .flat_map(|row| cpu_q4_affine_matvec(row, &quantized, &scales, &biases, output_rows))
            .collect();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.08,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn fused_q4_mlp_batch_matches_cpu_reference() {
        // This crosses the MPS prefill threshold while retaining the
        // decode-specific single-row assertion below.
        let batch_size = Q4_MPS_PREFILL_MIN_BATCH;
        let input_width = 64;
        let intermediate_width = 64;
        let output_width = 64;
        let input: Vec<f32> = (0..batch_size * input_width)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
            .collect();
        let gate_quantized: Vec<u8> = (0..intermediate_width * input_width)
            .map(|index| ((index * 5 + 1) % 16) as u8)
            .collect();
        let up_quantized: Vec<u8> = (0..intermediate_width * input_width)
            .map(|index| ((index * 11 + 3) % 16) as u8)
            .collect();
        let down_quantized: Vec<u8> = (0..output_width * intermediate_width)
            .map(|index| ((index * 13 + 7) % 16) as u8)
            .collect();
        let gate_scales = vec![f32_to_bf16(0.0625); intermediate_width];
        let gate_biases = vec![f32_to_bf16(-0.015625); intermediate_width];
        let up_scales = vec![f32_to_bf16(0.03125); intermediate_width];
        let up_biases = vec![f32_to_bf16(0.0078125); intermediate_width];
        let down_scales = vec![f32_to_bf16(0.125); output_width];
        let down_biases = vec![f32_to_bf16(-0.03125); output_width];
        let runtime = MetalRuntime::new().unwrap();
        let gate_weights = runtime
            .buffer_from_slice(&pack_q4(&gate_quantized))
            .unwrap();
        let up_weights = runtime.buffer_from_slice(&pack_q4(&up_quantized)).unwrap();
        let down_weights = runtime
            .buffer_from_slice(&pack_q4(&down_quantized))
            .unwrap();
        let gate_scale_buffer = runtime.buffer_from_slice(&gate_scales).unwrap();
        let gate_bias_buffer = runtime.buffer_from_slice(&gate_biases).unwrap();
        let up_scale_buffer = runtime.buffer_from_slice(&up_scales).unwrap();
        let up_bias_buffer = runtime.buffer_from_slice(&up_biases).unwrap();
        let down_scale_buffer = runtime.buffer_from_slice(&down_scales).unwrap();
        let down_bias_buffer = runtime.buffer_from_slice(&down_biases).unwrap();
        let actual = runtime
            .q4_affine_mlp_mapped_batch(
                &input,
                batch_size,
                &MappedQ4AffineJob::new(
                    &gate_weights,
                    0,
                    &gate_scale_buffer,
                    0,
                    &gate_bias_buffer,
                    0,
                    intermediate_width,
                    true,
                ),
                &MappedQ4AffineJob::new(
                    &up_weights,
                    0,
                    &up_scale_buffer,
                    0,
                    &up_bias_buffer,
                    0,
                    intermediate_width,
                    true,
                ),
                &MappedQ4AffineJob::new(
                    &down_weights,
                    0,
                    &down_scale_buffer,
                    0,
                    &down_bias_buffer,
                    0,
                    output_width,
                    true,
                ),
            )
            .unwrap();

        let mut expected = Vec::with_capacity(batch_size * output_width);
        for row in input.chunks_exact(input_width) {
            let gate = cpu_q4_affine_matvec(
                row,
                &gate_quantized,
                &gate_scales,
                &gate_biases,
                intermediate_width,
            );
            let up = cpu_q4_affine_matvec(
                row,
                &up_quantized,
                &up_scales,
                &up_biases,
                intermediate_width,
            );
            let swiglu: Vec<f32> = gate
                .into_iter()
                .zip(up)
                .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
                .collect();
            expected.extend(cpu_q4_affine_matvec(
                &swiglu,
                &down_quantized,
                &down_scales,
                &down_biases,
                output_width,
            ));
        }
        let prefill_tolerance = if runtime.mps_q4_prefill { 0.15 } else { 0.001 };
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < prefill_tolerance,
                "actual {actual}, expected {expected}"
            );
        }

        // Exercise the decode-specific matvec chain as well as the batched
        // prefill path above.
        let single_input = &input[..input_width];
        let single_actual = runtime
            .q4_affine_mlp_mapped_batch(
                single_input,
                1,
                &MappedQ4AffineJob::new(
                    &gate_weights,
                    0,
                    &gate_scale_buffer,
                    0,
                    &gate_bias_buffer,
                    0,
                    intermediate_width,
                    true,
                ),
                &MappedQ4AffineJob::new(
                    &up_weights,
                    0,
                    &up_scale_buffer,
                    0,
                    &up_bias_buffer,
                    0,
                    intermediate_width,
                    true,
                ),
                &MappedQ4AffineJob::new(
                    &down_weights,
                    0,
                    &down_scale_buffer,
                    0,
                    &down_bias_buffer,
                    0,
                    output_width,
                    true,
                ),
            )
            .unwrap();
        let gate = cpu_q4_affine_matvec(
            single_input,
            &gate_quantized,
            &gate_scales,
            &gate_biases,
            intermediate_width,
        );
        let up = cpu_q4_affine_matvec(
            single_input,
            &up_quantized,
            &up_scales,
            &up_biases,
            intermediate_width,
        );
        let swiglu: Vec<f32> = gate
            .into_iter()
            .zip(up)
            .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect();
        let single_expected = cpu_q4_affine_matvec(
            &swiglu,
            &down_quantized,
            &down_scales,
            &down_biases,
            output_width,
        );
        for (actual, expected) in single_actual.into_iter().zip(single_expected) {
            assert!(
                (actual - expected).abs() < 0.001,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn deltanet_step_matches_a_scalar_recurrence() {
        let config = DeltaNetConfig {
            key_heads: 1,
            value_heads: 1,
            key_head_dim: 2,
            value_head_dim: 2,
            conv_kernel_size: 1,
        };
        let qkv = vec![-0.5, 0.25, 0.75, -1.0, 0.2, -0.4];
        let z = vec![0.1, -0.2];
        let b = vec![0.3];
        let a = vec![-0.1];
        let a_log = vec![0.2];
        let dt_bias = vec![-0.3];
        let norm = vec![1.1, 0.9];
        let epsilon = 1e-6;
        let runtime = MetalRuntime::new().unwrap();
        let weights = runtime
            .create_deltanet_weights(config, &vec![1.0; qkv.len()], &a_log, &dt_bias, &norm)
            .unwrap();
        let mut state = runtime.create_deltanet_state(&weights).unwrap();
        let actual = runtime
            .deltanet_step(&weights, &mut state, &qkv, &z, &b, &a, epsilon)
            .unwrap();
        let expected = cpu_deltanet_step(&qkv, &z, &b, &a, &a_log, &dt_bias, &norm, epsilon);
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.001,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn gpu_decode_linear_layer_matches_the_segmented_path() {
        const HIDDEN: usize = 64;
        let epsilon = 1e-6;
        let runtime = MetalRuntime::new().unwrap();
        let make_q4_buffers = |output_rows: usize, seed: usize| {
            let quantized: Vec<u8> = (0..output_rows * HIDDEN)
                .map(|index| ((index * 7 + seed) % 16) as u8)
                .collect();
            let scales = (0..output_rows)
                .map(|index| f32_to_bf16(0.01171875 + (index % 3) as f32 * 0.00390625))
                .collect::<Vec<_>>();
            let biases = (0..output_rows)
                .map(|index| f32_to_bf16(-0.015625 + (index % 5) as f32 * 0.00390625))
                .collect::<Vec<_>>();
            (
                runtime.buffer_from_slice(&pack_q4(&quantized)).unwrap(),
                runtime.buffer_from_slice(&scales).unwrap(),
                runtime.buffer_from_slice(&biases).unwrap(),
            )
        };
        let config = DeltaNetConfig {
            key_heads: 1,
            value_heads: 1,
            key_head_dim: 64,
            value_head_dim: 64,
            conv_kernel_size: 1,
        };
        let channels = config.channels().unwrap();
        let (qkv_weights, qkv_scales, qkv_biases) = make_q4_buffers(channels, 1);
        let (z_weights, z_scales, z_biases) = make_q4_buffers(HIDDEN, 2);
        let (b_weights, b_scales, b_biases) = make_q4_buffers(1, 3);
        let (a_weights, a_scales, a_biases) = make_q4_buffers(1, 4);
        let (out_weights, out_scales, out_biases) = make_q4_buffers(HIDDEN, 5);
        let (gate_weights, gate_scales, gate_biases) = make_q4_buffers(HIDDEN, 6);
        let (up_weights, up_scales, up_biases) = make_q4_buffers(HIDDEN, 7);
        let (down_weights, down_scales, down_biases) = make_q4_buffers(HIDDEN, 8);
        let qkv_job = MappedQ4AffineJob::new(
            &qkv_weights,
            0,
            &qkv_scales,
            0,
            &qkv_biases,
            0,
            channels,
            true,
        );
        let z_job = MappedQ4AffineJob::new(&z_weights, 0, &z_scales, 0, &z_biases, 0, HIDDEN, true);
        let b_job = MappedQ4AffineJob::new(&b_weights, 0, &b_scales, 0, &b_biases, 0, 1, true);
        let a_job = MappedQ4AffineJob::new(&a_weights, 0, &a_scales, 0, &a_biases, 0, 1, true);
        let out_job = MappedQ4AffineJob::new(
            &out_weights,
            0,
            &out_scales,
            0,
            &out_biases,
            0,
            HIDDEN,
            true,
        );
        let gate_job = MappedQ4AffineJob::new(
            &gate_weights,
            0,
            &gate_scales,
            0,
            &gate_biases,
            0,
            HIDDEN,
            true,
        );
        let up_job =
            MappedQ4AffineJob::new(&up_weights, 0, &up_scales, 0, &up_biases, 0, HIDDEN, true);
        let down_job = MappedQ4AffineJob::new(
            &down_weights,
            0,
            &down_scales,
            0,
            &down_biases,
            0,
            HIDDEN,
            true,
        );
        let delta_weights = runtime
            .create_deltanet_weights(
                config,
                &(0..channels)
                    .map(|index| 0.75 + (index % 5) as f32 * 0.03125)
                    .collect::<Vec<_>>(),
                &[0.15],
                &[-0.2],
                &(0..HIDDEN)
                    .map(|index| 0.9 + (index % 7) as f32 * 0.015625)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let input_norm: Vec<f32> = (0..HIDDEN)
            .map(|index| 0.95 + (index % 5) as f32 * 0.015625)
            .collect();
        let post_norm: Vec<f32> = (0..HIDDEN)
            .map(|index| 0.9 + (index % 3) as f32 * 0.03125)
            .collect();
        let input: Vec<f32> = (0..HIDDEN)
            .map(|index| ((index % 19) as f32 - 9.0) * 0.0625)
            .collect();
        let rms = |values: &[f32], scale: &[f32]| {
            let mean = values.iter().map(|value| value * value).sum::<f32>() / HIDDEN as f32;
            let inverse = (mean + epsilon).sqrt().recip();
            values
                .iter()
                .zip(scale)
                .map(|(value, scale)| value * inverse * scale)
                .collect::<Vec<_>>()
        };

        let normalized = rms(&input, &input_norm);
        let mut projections = runtime
            .q4_affine_matvec_mapped_batch(&normalized, &[qkv_job, z_job, b_job, a_job])
            .unwrap();
        let qkv = projections.remove(0);
        let z = projections.remove(0);
        let b = projections.remove(0);
        let a = projections.remove(0);
        let mut segmented_state = runtime.create_deltanet_state(&delta_weights).unwrap();
        let delta = runtime
            .deltanet_step(
                &delta_weights,
                &mut segmented_state,
                &qkv,
                &z,
                &b,
                &a,
                epsilon,
            )
            .unwrap();
        let mut expected = input.clone();
        let mixed = runtime
            .q4_affine_matvec_mapped_batch(&delta, &[out_job])
            .unwrap()
            .remove(0);
        for (hidden, mixed) in expected.iter_mut().zip(mixed) {
            *hidden += mixed;
        }
        let normalized = rms(&expected, &post_norm);
        let mlp = runtime
            .q4_affine_mlp_mapped_batch(&normalized, 1, &gate_job, &up_job, &down_job)
            .unwrap();
        for (hidden, mlp) in expected.iter_mut().zip(mlp) {
            *hidden += mlp;
        }

        let input_norm_gpu = runtime.create_f32_buffer(&input_norm).unwrap();
        let post_norm_gpu = runtime.create_f32_buffer(&post_norm).unwrap();
        let mut decode = runtime.create_decode_state(HIDDEN).unwrap();
        runtime.write_decode_hidden(&mut decode, &input).unwrap();
        let fused_state = runtime.create_deltanet_state(&delta_weights).unwrap();
        runtime
            .decode_linear_layer(
                &mut decode,
                &input_norm_gpu,
                &post_norm_gpu,
                &qkv_job,
                &z_job,
                &b_job,
                &a_job,
                &out_job,
                &delta_weights,
                &fused_state,
                &gate_job,
                &up_job,
                &down_job,
                epsilon,
            )
            .unwrap();
        let actual = runtime.read_decode_hidden(&decode).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.003,
                "actual {actual}, expected {expected}"
            );
        }

        let step_first_state = runtime.create_deltanet_state(&delta_weights).unwrap();
        let step_second_state = runtime.create_deltanet_state(&delta_weights).unwrap();
        let mut stepped = runtime.create_decode_state(HIDDEN).unwrap();
        runtime.write_decode_hidden(&mut stepped, &input).unwrap();
        for delta_state in [&step_first_state, &step_second_state] {
            runtime
                .decode_linear_layer(
                    &mut stepped,
                    &input_norm_gpu,
                    &post_norm_gpu,
                    &qkv_job,
                    &z_job,
                    &b_job,
                    &a_job,
                    &out_job,
                    &delta_weights,
                    delta_state,
                    &gate_job,
                    &up_job,
                    &down_job,
                    epsilon,
                )
                .unwrap();
        }
        let expected_two_layers = runtime.read_decode_hidden(&stepped).unwrap();

        let batched_first_state = runtime.create_deltanet_state(&delta_weights).unwrap();
        let batched_second_state = runtime.create_deltanet_state(&delta_weights).unwrap();
        let mut batched = runtime.create_decode_state(HIDDEN).unwrap();
        runtime.write_decode_hidden(&mut batched, &input).unwrap();
        let first_layer = MetalDecodeLinearLayer::new(
            &input_norm_gpu,
            &post_norm_gpu,
            qkv_job,
            z_job,
            b_job,
            a_job,
            out_job,
            &delta_weights,
            &batched_first_state,
            gate_job,
            up_job,
            down_job,
        );
        let second_layer = MetalDecodeLinearLayer::new(
            &input_norm_gpu,
            &post_norm_gpu,
            qkv_job,
            z_job,
            b_job,
            a_job,
            out_job,
            &delta_weights,
            &batched_second_state,
            gate_job,
            up_job,
            down_job,
        );
        let mut graph = [
            MetalDecodeLayer::Linear(first_layer),
            MetalDecodeLayer::Linear(second_layer),
        ];
        runtime
            .decode_layers(&mut batched, &mut graph, epsilon)
            .unwrap();
        let actual_two_layers = runtime.read_decode_hidden(&batched).unwrap();
        for (actual, expected) in actual_two_layers.into_iter().zip(expected_two_layers) {
            assert!(
                (actual - expected).abs() < 0.003,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn deltanet_prefill_matches_steps_and_keeps_recurrent_state() {
        let config = DeltaNetConfig {
            key_heads: 1,
            value_heads: 1,
            key_head_dim: 2,
            value_head_dim: 2,
            conv_kernel_size: 2,
        };
        let qkv = vec![
            -0.5, 0.25, 0.75, -1.0, 0.2, -0.4, // token 0
            0.1, -0.7, -0.2, 0.6, 0.9, 0.3, // token 1
            0.8, -0.1, 0.4, -0.5, -0.6, 0.7, // token 2
        ];
        let z = vec![0.1, -0.2, -0.3, 0.4, 0.5, -0.6];
        let b = vec![0.3, -0.4, 0.2];
        let a = vec![-0.1, 0.25, -0.35];
        let conv_weight = vec![
            0.2, 0.8, -0.1, 0.7, 0.3, 0.6, -0.2, 0.9, 0.15, 0.85, -0.25, 0.75,
        ];
        let a_log = vec![0.2];
        let dt_bias = vec![-0.3];
        let norm = vec![1.1, 0.9];
        let epsilon = 1e-6;
        let runtime = MetalRuntime::new().unwrap();
        let weights = runtime
            .create_deltanet_weights(config, &conv_weight, &a_log, &dt_bias, &norm)
            .unwrap();
        let mut step_state = runtime.create_deltanet_state(&weights).unwrap();
        let mut prefill_state = runtime.create_deltanet_state(&weights).unwrap();
        let mut expected = Vec::new();
        for token in 0..3 {
            expected.extend(
                runtime
                    .deltanet_step(
                        &weights,
                        &mut step_state,
                        &qkv[token * 6..(token + 1) * 6],
                        &z[token * 2..(token + 1) * 2],
                        &b[token..token + 1],
                        &a[token..token + 1],
                        epsilon,
                    )
                    .unwrap(),
            );
        }
        let actual = runtime
            .deltanet_prefill(&weights, &mut prefill_state, &qkv, &z, &b, &a, 3, epsilon)
            .unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.001,
                "actual {actual}, expected {expected}"
            );
        }

        let next_qkv = [-0.35, 0.45, 0.55, -0.15, 0.25, 0.65];
        let next_z = [0.2, -0.45];
        let next_b = [0.15];
        let next_a = [-0.05];
        let expected_next = runtime
            .deltanet_step(
                &weights,
                &mut step_state,
                &next_qkv,
                &next_z,
                &next_b,
                &next_a,
                epsilon,
            )
            .unwrap();
        let actual_next = runtime
            .deltanet_step(
                &weights,
                &mut prefill_state,
                &next_qkv,
                &next_z,
                &next_b,
                &next_a,
                epsilon,
            )
            .unwrap();
        for (actual, expected) in actual_next.into_iter().zip(expected_next) {
            assert!(
                (actual - expected).abs() < 0.001,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn q8_gqa_attention_keeps_kv_on_the_gpu() {
        let runtime = MetalRuntime::new().unwrap();
        let mut state = runtime.create_q8_kv_state(1, 2).unwrap();
        let actual = runtime
            .gqa_attention_q8(
                &mut state,
                &[1.0, 0.0],
                &[0.0, 0.0],
                &[1.0, 0.0],
                &[2.0, -1.0],
                1,
            )
            .unwrap();
        assert_eq!(state.sequence_length, 1);
        assert!((actual[0] - 1.0).abs() < 0.001, "actual {:?}", actual);
        // Per-head int8 quantization rounds -1.0 to -64 with a 2/127 scale,
        // so the one-token result is approximately -0.503937.
        assert!((actual[1] + 0.5).abs() < 0.005, "actual {:?}", actual);
    }

    #[test]
    fn q8_gqa_prefill_matches_steps_and_keeps_kv_state() {
        let runtime = MetalRuntime::new().unwrap();
        let mut step_state = runtime.create_q8_kv_state(1, 2).unwrap();
        let mut prefill_state = runtime.create_q8_kv_state(1, 2).unwrap();
        let queries = vec![
            1.0, 0.0, 0.25, -0.5, // token 0, heads 0..1
            0.5, 0.5, -0.75, 0.25, // token 1
            -0.2, 0.8, 0.6, -0.4, // token 2
        ];
        let gates = vec![
            0.0, 0.0, 0.1, -0.1, // token 0
            -0.2, 0.2, 0.0, 0.0, // token 1
            0.3, -0.3, 0.15, 0.05, // token 2
        ];
        let keys = vec![1.0, 0.0, 0.4, 0.9, -0.7, 0.2];
        let values = vec![2.0, -1.0, 0.75, 1.25, -0.5, 0.4];
        let mut expected = Vec::new();
        for token in 0..3 {
            expected.extend(
                runtime
                    .gqa_attention_q8(
                        &mut step_state,
                        &queries[token * 4..(token + 1) * 4],
                        &gates[token * 4..(token + 1) * 4],
                        &keys[token * 2..(token + 1) * 2],
                        &values[token * 2..(token + 1) * 2],
                        2,
                    )
                    .unwrap(),
            );
        }
        let actual = runtime
            .gqa_attention_q8_prefill(&mut prefill_state, &queries, &gates, &keys, &values, 2, 3)
            .unwrap();
        assert_eq!(prefill_state.sequence_length, 3);
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.005,
                "actual {actual}, expected {expected}"
            );
        }

        let next_query = [0.9, -0.1, -0.3, 0.7];
        let next_gate = [0.05, -0.05, 0.2, -0.2];
        let next_key = [0.2, -0.6];
        let next_value = [1.1, -0.8];
        let expected_next = runtime
            .gqa_attention_q8(
                &mut step_state,
                &next_query,
                &next_gate,
                &next_key,
                &next_value,
                2,
            )
            .unwrap();
        let actual_next = runtime
            .gqa_attention_q8(
                &mut prefill_state,
                &next_query,
                &next_gate,
                &next_key,
                &next_value,
                2,
            )
            .unwrap();
        for (actual, expected) in actual_next.into_iter().zip(expected_next) {
            assert!(
                (actual - expected).abs() < 0.005,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn q4_affine_matvec_rejects_incomplete_group() {
        let error = MatvecShape::validate(&[0.0; 63], &[], &[], &[], 1).unwrap_err();
        assert!(matches!(error, MetalRuntimeError::InputNotGrouped { .. }));
    }

    #[test]
    fn bf16_gemm_reads_a_mapped_row_major_matrix() {
        let input = vec![1.0, -2.0, 0.5, 3.0, 0.25, -1.0];
        let weights = [
            0.5, -1.0, 2.0, // output column 0
            -0.25, 1.5, 0.75, // output column 1
        ];
        let mut bytes = Vec::with_capacity(weights.len() * 2);
        for value in weights {
            bytes.extend(f32_to_bf16(value).to_le_bytes());
        }
        let runtime = MetalRuntime::new().unwrap();
        let buffer = runtime.buffer_from_slice(&bytes).unwrap();
        let actual = runtime.bf16_gemm_mapped(&input, &buffer, 0, 3, 2).unwrap();
        let expected = vec![3.5, -2.875, -0.75, -1.125];
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.02,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn vision_attention_matches_cpu_reference() {
        let sequence_length = 3;
        let num_heads = 2;
        let head_dim = 2;
        let queries = vec![
            1.0, 0.0, 0.5, 0.5, // token 0
            0.0, 1.0, 0.25, 0.75, // token 1
            1.0, 1.0, -0.5, 0.25, // token 2
        ];
        let keys = vec![
            1.0, 0.0, 0.0, 1.0, // token 0
            0.0, 1.0, 1.0, 0.0, // token 1
            1.0, 1.0, 0.5, 0.5, // token 2
        ];
        let values = vec![
            1.0, 2.0, 3.0, 4.0, // token 0
            5.0, 6.0, 7.0, 8.0, // token 1
            9.0, 10.0, 11.0, 12.0, // token 2
        ];
        let runtime = MetalRuntime::new().unwrap();
        let actual = runtime
            .vision_attention(
                &queries,
                &keys,
                &values,
                sequence_length,
                num_heads,
                head_dim,
            )
            .unwrap();
        let expected = cpu_vision_attention(
            &queries,
            &keys,
            &values,
            sequence_length,
            num_heads,
            head_dim,
        );
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.002,
                "actual {actual}, expected {expected}"
            );
        }
    }

    fn pack_q4(values: &[u8]) -> Vec<u32> {
        assert_eq!(values.len() % VALUES_PER_PACKED_WORD, 0);
        values
            .chunks_exact(VALUES_PER_PACKED_WORD)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0_u32, |packed, (index, value)| {
                        packed | (u32::from(*value) << (index * QUANT_BITS))
                    })
            })
            .collect()
    }

    fn cpu_q4_affine_matvec(
        input: &[f32],
        quantized: &[u8],
        scales: &[u16],
        biases: &[u16],
        output_rows: usize,
    ) -> Vec<f32> {
        let groups_per_row = input.len() / AFFINE_GROUP_SIZE;
        (0..output_rows)
            .map(|row| {
                (0..input.len()).fold(0.0_f32, |total, column| {
                    let group = column / AFFINE_GROUP_SIZE;
                    let parameters = row * groups_per_row + group;
                    let scale = bf16_to_f32(scales[parameters]);
                    let bias = bf16_to_f32(biases[parameters]);
                    let weight = f32::from(quantized[row * input.len() + column]) * scale + bias;
                    total + input[column] * weight
                })
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_deltanet_step(
        qkv: &[f32],
        z: &[f32],
        b: &[f32],
        a: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        norm: &[f32],
        epsilon: f32,
    ) -> Vec<f32> {
        let convolved: Vec<f32> = qkv.iter().copied().map(cpu_silu).collect();
        let key_dim = 2;
        let mut query = convolved[..key_dim].to_vec();
        let mut key = convolved[key_dim..key_dim * 2].to_vec();
        let inverse_head_scale = (key_dim as f32).sqrt().recip();
        let query_rms = (query.iter().map(|value| value * value).sum::<f32>() / key_dim as f32
            + epsilon)
            .sqrt()
            .recip();
        let key_rms = (key.iter().map(|value| value * value).sum::<f32>() / key_dim as f32
            + epsilon)
            .sqrt()
            .recip();
        for value in &mut query {
            *value *= query_rms * inverse_head_scale * inverse_head_scale;
        }
        for value in &mut key {
            *value *= key_rms * inverse_head_scale;
        }
        let beta = cpu_sigmoid(b[0]);
        let decay = (-a_log[0].exp() * cpu_softplus(a[0] + dt_bias[0])).exp();
        let mut recurrent = vec![0.0; key_dim * z.len()];
        let mut output = vec![0.0; z.len()];
        for value_index in 0..z.len() {
            let state = &mut recurrent[value_index * key_dim..(value_index + 1) * key_dim];
            let kv_mem = state
                .iter_mut()
                .zip(&key)
                .map(|(state, key)| {
                    *state *= decay;
                    *state * key
                })
                .sum::<f32>();
            let delta = (convolved[key_dim * 2 + value_index] - kv_mem) * beta;
            output[value_index] = state
                .iter_mut()
                .zip(query.iter().zip(&key))
                .map(|(state, (query, key))| {
                    *state += key * delta;
                    *state * query
                })
                .sum();
        }
        let output_rms =
            (output.iter().map(|value| value * value).sum::<f32>() / output.len() as f32 + epsilon)
                .sqrt()
                .recip();
        output
            .into_iter()
            .zip(z)
            .zip(norm)
            .map(|((value, z), norm)| value * output_rms * norm * cpu_silu(*z))
            .collect()
    }

    fn cpu_sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn cpu_silu(value: f32) -> f32 {
        value * cpu_sigmoid(value)
    }

    fn cpu_softplus(value: f32) -> f32 {
        if value > 20.0 {
            value
        } else if value < -20.0 {
            value.exp()
        } else {
            (1.0 + value.exp()).ln()
        }
    }

    fn cpu_vision_attention(
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        sequence_length: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; sequence_length * num_heads * head_dim];
        let scale = (head_dim as f32).sqrt().recip();
        for query in 0..sequence_length {
            for head in 0..num_heads {
                let query_offset = (query * num_heads + head) * head_dim;
                let mut scores = Vec::with_capacity(sequence_length);
                for key in 0..sequence_length {
                    let key_offset = (key * num_heads + head) * head_dim;
                    let score = (0..head_dim)
                        .map(|dimension| {
                            queries[query_offset + dimension] * keys[key_offset + dimension]
                        })
                        .sum::<f32>()
                        * scale;
                    scores.push(score);
                }
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let normalizer = scores
                    .iter()
                    .map(|score| (*score - max_score).exp())
                    .sum::<f32>();
                for dimension in 0..head_dim {
                    let output_offset = (query * num_heads + head) * head_dim + dimension;
                    output[output_offset] = scores
                        .iter()
                        .enumerate()
                        .map(|(key, score)| {
                            let value_offset = (key * num_heads + head) * head_dim + dimension;
                            (*score - max_score).exp() / normalizer * values[value_offset]
                        })
                        .sum();
                }
            }
        }
        output
    }

    fn f32_to_bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
        (rounded >> 16) as u16
    }

    fn bf16_to_f32(value: u16) -> f32 {
        f32::from_bits(u32::from(value) << 16)
    }
}
