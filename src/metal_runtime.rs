use crate::metal::EMBEDDED_LIBRARY;
use crate::model::MlxWeightStore;
use metal::{ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use std::error::Error;
use std::fmt;
use std::mem::size_of;

const QUANT_BITS: usize = 4;
const VALUES_PER_PACKED_WORD: usize = 32 / QUANT_BITS;
const AFFINE_GROUP_SIZE: usize = 64;
const THREADS_PER_THREADGROUP: u64 = 256;

pub struct MetalRuntime {
    device: Device,
    command_queue: metal::CommandQueue,
    q4_affine_matvec: ComputePipelineState,
    q4_affine_matvec_unaligned: ComputePipelineState,
    bf16_gemm: ComputePipelineState,
    vision_attention_scores: ComputePipelineState,
    vision_attention_values: ComputePipelineState,
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
        let function = library
            .get_function("qwen38_q4_affine_matvec", None)
            .map_err(MetalRuntimeError::Function)?;
        let q4_affine_matvec = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(MetalRuntimeError::Pipeline)?;
        let unaligned_function = library
            .get_function("qwen38_q4_affine_matvec_unaligned", None)
            .map_err(MetalRuntimeError::Function)?;
        let q4_affine_matvec_unaligned = device
            .new_compute_pipeline_state_with_function(&unaligned_function)
            .map_err(MetalRuntimeError::Pipeline)?;
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

        if q4_affine_matvec.max_total_threads_per_threadgroup() < THREADS_PER_THREADGROUP
            || q4_affine_matvec_unaligned.max_total_threads_per_threadgroup()
                < THREADS_PER_THREADGROUP
        {
            return Err(MetalRuntimeError::UnsupportedThreadgroupLimit {
                available: q4_affine_matvec
                    .max_total_threads_per_threadgroup()
                    .min(q4_affine_matvec_unaligned.max_total_threads_per_threadgroup()),
                required: THREADS_PER_THREADGROUP,
            });
        }

        Ok(Self {
            command_queue: device.new_command_queue(),
            device,
            q4_affine_matvec,
            q4_affine_matvec_unaligned,
            bf16_gemm,
            vision_attention_scores,
            vision_attention_values,
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
        let metal_output_rows = u64::try_from(output_rows)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("output row count"))?;
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.q4_affine_matvec);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weights), weight_offset);
        encoder.set_buffer(2, Some(scales), scale_offset);
        encoder.set_buffer(3, Some(biases), bias_offset);
        encoder.set_buffer(4, Some(&output_buffer), 0);
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&words_per_row as *const u32).cast(),
        );
        encoder.set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        encoder.set_threadgroup_memory_length(1, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(metal_output_rows, 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
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
        let metal_output_rows = u64::try_from(output_rows)
            .map_err(|_| MetalRuntimeError::DimensionOverflow("output row count"))?;
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.q4_affine_matvec_unaligned);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weights), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_buffer(3, Some(biases), 0);
        encoder.set_buffer(4, Some(&output_buffer), 0);
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&words_per_row as *const u32).cast(),
        );
        encoder.set_bytes(
            6,
            size_of::<u64>() as u64,
            (&weight_offset as *const u64).cast(),
        );
        encoder.set_bytes(
            7,
            size_of::<u64>() as u64,
            (&scale_offset as *const u64).cast(),
        );
        encoder.set_bytes(
            8,
            size_of::<u64>() as u64,
            (&bias_offset as *const u64).cast(),
        );
        encoder.set_threadgroup_memory_length(0, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        encoder.set_threadgroup_memory_length(1, THREADS_PER_THREADGROUP * size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(metal_output_rows, 1, 1),
            MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
        );
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
    if input.is_empty() || output_rows == 0 {
        return Err(MetalRuntimeError::EmptyDimension);
    }
    if input.len() % AFFINE_GROUP_SIZE != 0 {
        return Err(MetalRuntimeError::InputNotGrouped {
            input_elements: input.len(),
            group_size: AFFINE_GROUP_SIZE,
        });
    }
    if weight_offset > weights.length()
        || scale_offset > scales.length()
        || bias_offset > biases.length()
    {
        return Err(MetalRuntimeError::InvalidBufferOffset);
    }

    let words_per_row = input.len() / VALUES_PER_PACKED_WORD;
    let required_weight_bytes = checked_byte_len::<u32>(
        output_rows
            .checked_mul(words_per_row)
            .ok_or(MetalRuntimeError::DimensionOverflow("packed weight count"))?,
    )?;
    let groups_per_row = input.len() / AFFINE_GROUP_SIZE;
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
    EmptyDimension,
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
            Self::EmptyDimension => write!(formatter, "matrix dimensions must be greater than zero"),
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
            Self::CommandFailed => write!(formatter, "Metal command buffer failed"),
        }
    }
}

impl Error for MetalRuntimeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_affine_matvec_matches_cpu_reference() {
        let input: Vec<f32> = (0..128)
            .map(|index| ((index % 17) as f32 - 8.0) * 0.125)
            .collect();
        let output_rows = 3;
        let quantized: Vec<u8> = (0..output_rows * input.len())
            .map(|index| ((index * 7 + 3) % 16) as u8)
            .collect();
        let packed_weights = pack_q4(&quantized);
        let scale_values = [0.125, -0.0625, 0.25, 0.03125, -0.125, 0.0625];
        let bias_values = [0.03125, -0.015625, 0.0625, 0.125, -0.03125, 0.0];
        let scales: Vec<u16> = scale_values.into_iter().map(f32_to_bf16).collect();
        let biases: Vec<u16> = bias_values.into_iter().map(f32_to_bf16).collect();

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
    fn q4_affine_matvec_reads_unaligned_mapped_data() {
        let input: Vec<f32> = (0..128)
            .map(|index| ((index % 19) as f32 - 9.0) * 0.0625)
            .collect();
        let output_rows = 3;
        let quantized: Vec<u8> = (0..output_rows * input.len())
            .map(|index| ((index * 11 + 5) % 16) as u8)
            .collect();
        let packed_weights = pack_q4(&quantized);
        let scales = vec![f32_to_bf16(0.125); output_rows * 2];
        let biases = vec![f32_to_bf16(-0.03125); output_rows * 2];

        let mut weight_bytes = vec![0_u8];
        for value in &packed_weights {
            weight_bytes.extend(value.to_le_bytes());
        }
        let mut scale_bytes = vec![0_u8];
        for value in &scales {
            scale_bytes.extend(value.to_le_bytes());
        }
        let mut bias_bytes = vec![0_u8];
        for value in &biases {
            bias_bytes.extend(value.to_le_bytes());
        }

        let runtime = MetalRuntime::new().unwrap();
        let weights = runtime.buffer_from_slice(&weight_bytes).unwrap();
        let scale_buffer = runtime.buffer_from_slice(&scale_bytes).unwrap();
        let bias_buffer = runtime.buffer_from_slice(&bias_bytes).unwrap();
        let actual = runtime
            .q4_affine_matvec_mapped_unaligned(
                &input,
                &weights,
                1,
                &scale_buffer,
                1,
                &bias_buffer,
                1,
                output_rows,
            )
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
