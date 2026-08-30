#include <metal_stdlib>

using namespace metal;

// This library is built before the Rust binary is linked. Future kernels keep
// Q4e dequantization, paged Q8 KV attention, Gated DeltaNet, and MTP verify in
// this precompiled library rather than compiling source at runtime.
kernel void qwen38_warmup(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index == 0) {
        output[0] = input[0];
    }
}

inline float qwen38_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

// The MLX Q4 affine layout stores eight low-to-high 4-bit values in each u32.
// Each 64-element group has one BF16 scale and bias. This computes
// sum(x * q) * scale + sum(x) * bias without materializing dequantized weights.
kernel void qwen38_q4_affine_matvec(
    device const float* input [[buffer(0)]],
    device const uint* packed_weights [[buffer(1)]],
    device const ushort* scales [[buffer(2)]],
    device const ushort* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& words_per_row [[buffer(5)]],
    threadgroup float* partial_dot [[threadgroup(0)]],
    threadgroup float* partial_sum [[threadgroup(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_position_in_threadgroup]]) {
    constexpr uint words_per_group = 8;
    float dot_sum = 0.0f;
    float input_sum = 0.0f;
    const uint row_weight_offset = row * words_per_row;
    const uint row_group_offset = row * (words_per_row / words_per_group);

    for (uint word_index = thread_index; word_index < words_per_row; word_index += 256) {
        const uint packed = packed_weights[row_weight_offset + word_index];
        const float scale = qwen38_bf16_to_float(scales[row_group_offset + word_index / words_per_group]);
        const float bias = qwen38_bf16_to_float(biases[row_group_offset + word_index / words_per_group]);
        const uint input_offset = word_index * 8;

        for (uint nibble_index = 0; nibble_index < 8; ++nibble_index) {
            const float value = input[input_offset + nibble_index];
            const float quantized = float((packed >> (nibble_index * 4)) & 0xFu);
            dot_sum += value * quantized * scale;
            input_sum += value * bias;
        }
    }

    partial_dot[thread_index] = dot_sum;
    partial_sum[thread_index] = input_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride /= 2) {
        if (thread_index < stride) {
            partial_dot[thread_index] += partial_dot[thread_index + stride];
            partial_sum[thread_index] += partial_sum[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        output[row] = partial_dot[0] + partial_sum[0];
    }
}

// Safetensors permits a header whose length leaves the tensor data region
// unaligned. Binding the whole mapped shard at offset zero and decoding bytes
// here keeps that legal layout zero-copy on Metal.
kernel void qwen38_q4_affine_matvec_unaligned(
    device const float* input [[buffer(0)]],
    device const uchar* packed_weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const uchar* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& words_per_row [[buffer(5)]],
    constant ulong& weight_byte_offset [[buffer(6)]],
    constant ulong& scale_byte_offset [[buffer(7)]],
    constant ulong& bias_byte_offset [[buffer(8)]],
    threadgroup float* partial_dot [[threadgroup(0)]],
    threadgroup float* partial_sum [[threadgroup(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_position_in_threadgroup]]) {
    constexpr uint words_per_group = 8;
    float dot_sum = 0.0f;
    float input_sum = 0.0f;
    const ulong row_weight_offset = weight_byte_offset + ulong(row) * ulong(words_per_row) * 4ul;
    const ulong row_group_offset = scale_byte_offset
        + ulong(row) * ulong(words_per_row / words_per_group) * 2ul;
    const ulong row_bias_group_offset = bias_byte_offset
        + ulong(row) * ulong(words_per_row / words_per_group) * 2ul;

    for (uint word_index = thread_index; word_index < words_per_row; word_index += 256) {
        const ulong weight_address = row_weight_offset + ulong(word_index) * 4ul;
        const uint packed = uint(packed_weights[weight_address])
            | (uint(packed_weights[weight_address + 1ul]) << 8)
            | (uint(packed_weights[weight_address + 2ul]) << 16)
            | (uint(packed_weights[weight_address + 3ul]) << 24);
        const ulong group_address = row_group_offset + ulong(word_index / words_per_group) * 2ul;
        const ulong bias_group_address = row_bias_group_offset
            + ulong(word_index / words_per_group) * 2ul;
        const ushort scale_bits = ushort(scales[group_address])
            | (ushort(scales[group_address + 1ul]) << 8);
        const ushort bias_bits = ushort(biases[bias_group_address])
            | (ushort(biases[bias_group_address + 1ul]) << 8);
        const float scale = qwen38_bf16_to_float(scale_bits);
        const float bias = qwen38_bf16_to_float(bias_bits);
        const uint input_offset = word_index * 8;

        for (uint nibble_index = 0; nibble_index < 8; ++nibble_index) {
            const float value = input[input_offset + nibble_index];
            const float quantized = float((packed >> (nibble_index * 4)) & 0xFu);
            dot_sum += value * quantized * scale;
            input_sum += value * bias;
        }
    }

    partial_dot[thread_index] = dot_sum;
    partial_sum[thread_index] = input_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride /= 2) {
        if (thread_index < stride) {
            partial_dot[thread_index] += partial_dot[thread_index + stride];
            partial_sum[thread_index] += partial_sum[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        output[row] = partial_dot[0] + partial_sum[0];
    }
}

// Prefill tiles the activation matrix rather than launching one threadgroup
// for every output element. Each group computes eight prompt rows by 32
// output rows and stages a 64-element affine group of activations and packed
// Q4 weights. This keeps the mapped MLX layout zero-copy while making the
// long-prompt path a real tiled GEMM instead of a collection of matvecs.
#define QWEN38_PREFILL_BATCH_TILE 8u
#define QWEN38_PREFILL_OUTPUT_TILE 32u
#define QWEN38_PREFILL_AFFINE_TILE 64u
#define QWEN38_PREFILL_WORDS_PER_AFFINE_TILE 8u

kernel void qwen38_q4_affine_matmul(
    device const float* input [[buffer(0)]],
    device const uint* packed_weights [[buffer(1)]],
    device const ushort* scales [[buffer(2)]],
    device const ushort* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& words_per_row [[buffer(5)]],
    constant uint& output_rows [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    threadgroup float* input_tile [[threadgroup(0)]],
    threadgroup uint* packed_tile [[threadgroup(1)]],
    threadgroup float* affine_tile [[threadgroup(2)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint local_batch = thread_index / QWEN38_PREFILL_OUTPUT_TILE;
    const uint local_row = thread_index % QWEN38_PREFILL_OUTPUT_TILE;
    const uint batch = tile.y * QWEN38_PREFILL_BATCH_TILE + local_batch;
    const uint row = tile.x * QWEN38_PREFILL_OUTPUT_TILE + local_row;
    const uint input_width = words_per_row * 8;
    const uint affine_groups_per_row = words_per_row / QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
    threadgroup float* scale_tile = affine_tile;
    threadgroup float* bias_tile = affine_tile + QWEN38_PREFILL_OUTPUT_TILE;
    float accumulator = 0.0f;

    for (uint affine_group = 0; affine_group < affine_groups_per_row; ++affine_group) {
        for (uint index = thread_index;
             index < QWEN38_PREFILL_BATCH_TILE * QWEN38_PREFILL_AFFINE_TILE;
             index += 256) {
            const uint staged_batch = tile.y * QWEN38_PREFILL_BATCH_TILE
                + index / QWEN38_PREFILL_AFFINE_TILE;
            const uint staged_element = affine_group * QWEN38_PREFILL_AFFINE_TILE
                + index % QWEN38_PREFILL_AFFINE_TILE;
            input_tile[index] = staged_batch < batch_size
                ? input[staged_batch * input_width + staged_element]
                : 0.0f;
        }
        const uint packed_row = thread_index / QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
        const uint packed_word = thread_index % QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
        const uint staged_row = tile.x * QWEN38_PREFILL_OUTPUT_TILE + packed_row;
        packed_tile[thread_index] = staged_row < output_rows
            ? packed_weights[staged_row * words_per_row
                + affine_group * QWEN38_PREFILL_WORDS_PER_AFFINE_TILE + packed_word]
            : 0u;
        if (thread_index < QWEN38_PREFILL_OUTPUT_TILE) {
            const uint parameter_row = tile.x * QWEN38_PREFILL_OUTPUT_TILE + thread_index;
            const uint parameter_index = parameter_row * affine_groups_per_row + affine_group;
            scale_tile[thread_index] = parameter_row < output_rows
                ? qwen38_bf16_to_float(scales[parameter_index])
                : 0.0f;
            bias_tile[thread_index] = parameter_row < output_rows
                ? qwen38_bf16_to_float(biases[parameter_index])
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (batch < batch_size && row < output_rows) {
            const float scale = scale_tile[local_row];
            const float bias = bias_tile[local_row];
            const uint packed_base = local_row * QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
            const uint input_base = local_batch * QWEN38_PREFILL_AFFINE_TILE;
            for (uint word = 0; word < QWEN38_PREFILL_WORDS_PER_AFFINE_TILE; ++word) {
                const uint packed = packed_tile[packed_base + word];
                for (uint nibble = 0; nibble < 8; ++nibble) {
                    const float value = input_tile[input_base + word * 8 + nibble];
                    const float quantized = float((packed >> (nibble * 4)) & 0xFu);
                    accumulator += value * quantized * scale;
                    accumulator += value * bias;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (batch < batch_size && row < output_rows) {
        output[batch * output_rows + row] = accumulator;
    }
}

kernel void qwen38_q4_affine_matmul_unaligned(
    device const float* input [[buffer(0)]],
    device const uchar* packed_weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const uchar* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& words_per_row [[buffer(5)]],
    constant ulong& weight_byte_offset [[buffer(6)]],
    constant ulong& scale_byte_offset [[buffer(7)]],
    constant ulong& bias_byte_offset [[buffer(8)]],
    constant uint& output_rows [[buffer(9)]],
    constant uint& batch_size [[buffer(10)]],
    threadgroup float* input_tile [[threadgroup(0)]],
    threadgroup uint* packed_tile [[threadgroup(1)]],
    threadgroup float* affine_tile [[threadgroup(2)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint local_batch = thread_index / QWEN38_PREFILL_OUTPUT_TILE;
    const uint local_row = thread_index % QWEN38_PREFILL_OUTPUT_TILE;
    const uint batch = tile.y * QWEN38_PREFILL_BATCH_TILE + local_batch;
    const uint row = tile.x * QWEN38_PREFILL_OUTPUT_TILE + local_row;
    const uint input_width = words_per_row * 8;
    const uint affine_groups_per_row = words_per_row / QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
    threadgroup float* scale_tile = affine_tile;
    threadgroup float* bias_tile = affine_tile + QWEN38_PREFILL_OUTPUT_TILE;
    float accumulator = 0.0f;

    for (uint affine_group = 0; affine_group < affine_groups_per_row; ++affine_group) {
        for (uint index = thread_index;
             index < QWEN38_PREFILL_BATCH_TILE * QWEN38_PREFILL_AFFINE_TILE;
             index += 256) {
            const uint staged_batch = tile.y * QWEN38_PREFILL_BATCH_TILE
                + index / QWEN38_PREFILL_AFFINE_TILE;
            const uint staged_element = affine_group * QWEN38_PREFILL_AFFINE_TILE
                + index % QWEN38_PREFILL_AFFINE_TILE;
            input_tile[index] = staged_batch < batch_size
                ? input[staged_batch * input_width + staged_element]
                : 0.0f;
        }
        const uint packed_row = thread_index / QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
        const uint packed_word = thread_index % QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
        const uint staged_row = tile.x * QWEN38_PREFILL_OUTPUT_TILE + packed_row;
        if (staged_row < output_rows) {
            const ulong weight_address = weight_byte_offset
                + (ulong(staged_row) * ulong(words_per_row)
                    + ulong(affine_group * QWEN38_PREFILL_WORDS_PER_AFFINE_TILE + packed_word))
                    * 4ul;
            const uint packed = uint(packed_weights[weight_address])
                | (uint(packed_weights[weight_address + 1ul]) << 8)
                | (uint(packed_weights[weight_address + 2ul]) << 16)
                | (uint(packed_weights[weight_address + 3ul]) << 24);
            packed_tile[thread_index] = packed;
        } else {
            packed_tile[thread_index] = 0u;
        }
        if (thread_index < QWEN38_PREFILL_OUTPUT_TILE) {
            const uint parameter_row = tile.x * QWEN38_PREFILL_OUTPUT_TILE + thread_index;
            const ulong parameter_index = ulong(parameter_row * affine_groups_per_row + affine_group) * 2ul;
            if (parameter_row < output_rows) {
                const ulong scale_address = scale_byte_offset + parameter_index;
                const ulong bias_address = bias_byte_offset + parameter_index;
                const ushort scale_bits = ushort(scales[scale_address])
                    | (ushort(scales[scale_address + 1ul]) << 8);
                const ushort bias_bits = ushort(biases[bias_address])
                    | (ushort(biases[bias_address + 1ul]) << 8);
                scale_tile[thread_index] = qwen38_bf16_to_float(scale_bits);
                bias_tile[thread_index] = qwen38_bf16_to_float(bias_bits);
            } else {
                scale_tile[thread_index] = 0.0f;
                bias_tile[thread_index] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (batch < batch_size && row < output_rows) {
            const float scale = scale_tile[local_row];
            const float bias = bias_tile[local_row];
            const uint packed_base = local_row * QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
            const uint input_base = local_batch * QWEN38_PREFILL_AFFINE_TILE;
            for (uint word = 0; word < QWEN38_PREFILL_WORDS_PER_AFFINE_TILE; ++word) {
                const uint packed = packed_tile[packed_base + word];
                for (uint nibble = 0; nibble < 8; ++nibble) {
                    const float value = input_tile[input_base + word * 8 + nibble];
                    const float quantized = float((packed >> (nibble * 4)) & 0xFu);
                    accumulator += value * quantized * scale;
                    accumulator += value * bias;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (batch < batch_size && row < output_rows) {
        output[batch * output_rows + row] = accumulator;
    }
}

inline float qwen38_bf16_at(device const uchar* values, ulong byte_offset) {
    const ushort bits = ushort(values[byte_offset]) | (ushort(values[byte_offset + 1ul]) << 8);
    return qwen38_bf16_to_float(bits);
}

// Vision weights are BF16 rather than Q4. This deliberately keeps the
// safetensors mapping zero-copy: only the small activation matrices move over
// the CPU/GPU boundary. Each thread computes one output element; vision calls
// use enough independent rows and columns to keep all GPU cores occupied.
kernel void qwen38_bf16_gemm(
    device const float* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& input_rows [[buffer(4)]],
    constant uint& input_columns [[buffer(5)]],
    constant uint& output_columns [[buffer(6)]],
    uint2 index [[thread_position_in_grid]]) {
    const uint column = index.x;
    const uint row = index.y;
    if (row >= input_rows || column >= output_columns) {
        return;
    }

    float sum = 0.0f;
    const uint input_base = row * input_columns;
    const ulong weight_base = weight_byte_offset
        + ulong(column) * ulong(input_columns) * 2ul;
    for (uint element = 0; element < input_columns; ++element) {
        sum += input[input_base + element]
            * qwen38_bf16_at(weights, weight_base + ulong(element) * 2ul);
    }
    output[row * output_columns + column] = sum;
}

// A single threadgroup owns one (query, head) pair. It first normalizes the
// complete score row and writes it to a temporary matrix shared by the value
// projection kernel below. The visual encoder is bidirectional, unlike the
// causal language attention path.
kernel void qwen38_vision_attention_scores(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& sequence_length [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    threadgroup float* values [[threadgroup(0)]],
    uint2 query_head [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint query = query_head.x;
    const uint head = query_head.y;
    if (query >= sequence_length || head >= num_heads) {
        return;
    }

    const uint score_base = (head * sequence_length + query) * sequence_length;
    const uint query_base = (query * num_heads + head) * head_dim;
    float local_max = -INFINITY;
    for (uint key = thread_index; key < sequence_length; key += 256) {
        const uint key_base = (key * num_heads + head) * head_dim;
        float score = 0.0f;
        for (uint dimension = 0; dimension < head_dim; ++dimension) {
            score += queries[query_base + dimension] * keys[key_base + dimension];
        }
        score = score * rsqrt(float(head_dim));
        scores[score_base + key] = score;
        local_max = max(local_max, score);
    }
    values[thread_index] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            values[thread_index] = max(values[thread_index], values[thread_index + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = values[0];
    float local_sum = 0.0f;
    for (uint key = thread_index; key < sequence_length; key += 256) {
        const float exp_score = exp(scores[score_base + key] - max_score);
        scores[score_base + key] = exp_score;
        local_sum += exp_score;
    }
    values[thread_index] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            values[thread_index] += values[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint key = thread_index; key < sequence_length; key += 256) {
        scores[score_base + key] /= max(values[0], FLT_MIN);
    }
}

kernel void qwen38_vision_attention_values(
    device const float* scores [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& sequence_length [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint2 query_head [[threadgroup_position_in_grid]],
    uint dimension [[thread_index_in_threadgroup]]) {
    const uint query = query_head.x;
    const uint head = query_head.y;
    if (query >= sequence_length || head >= num_heads || dimension >= head_dim) {
        return;
    }

    float sum = 0.0f;
    const uint score_base = (head * sequence_length + query) * sequence_length;
    for (uint key = 0; key < sequence_length; ++key) {
        const uint value_offset = (key * num_heads + head) * head_dim + dimension;
        sum += scores[score_base + key] * values[value_offset];
    }
    output[(query * num_heads + head) * head_dim + dimension] = sum;
}

inline float qwen38_sigmoid(float value) {
    return 1.0f / (1.0f + exp(-value));
}

inline float qwen38_silu(float value) {
    return value * qwen38_sigmoid(value);
}

// Gate and up projections are produced by separate Q4 kernels, then combined
// without returning either intermediate to the CPU before the down projection.
kernel void qwen38_swiglu_rows(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        output[index] = qwen38_silu(gate[index]) * up[index];
    }
}

inline float qwen38_softplus(float value) {
    if (value > 20.0f) {
        return value;
    }
    if (value < -20.0f) {
        return exp(value);
    }
    return log(1.0f + exp(value));
}

// DeltaNet first applies its depthwise causal convolution and keeps its short
// history on the GPU. Every channel is independent at this stage.
kernel void qwen38_deltanet_conv(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* history [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= channels) {
        return;
    }
    const uint weight_base = channel * kernel_size;
    const uint history_width = kernel_size - 1;
    const uint history_base = channel * history_width;
    float sum = weights[weight_base + history_width] * input[channel];
    for (uint tap = 0; tap < history_width; ++tap) {
        sum += weights[weight_base + tap] * history[history_base + tap];
    }
    output[channel] = qwen38_silu(sum);
    if (history_width > 0) {
        for (uint tap = 0; tap + 1 < history_width; ++tap) {
            history[history_base + tap] = history[history_base + tap + 1];
        }
        history[history_base + history_width - 1] = input[channel];
    }
}

// Qwen's linear-attention q/k vectors are normalized headwise after the
// convolution. The normalized values replace the q/k portion in-place.
kernel void qwen38_deltanet_prepare(
    device float* convolved [[buffer(0)]],
    constant uint& key_heads [[buffer(1)]],
    constant uint& key_head_dim [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    threadgroup float* query_partial [[threadgroup(0)]],
    threadgroup float* key_partial [[threadgroup(1)]],
    uint head [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    if (head >= key_heads) {
        return;
    }
    const uint key_elements = key_heads * key_head_dim;
    const uint query_offset = head * key_head_dim;
    const uint key_offset = key_elements + query_offset;
    float query = 0.0f;
    float key = 0.0f;
    if (thread_index < key_head_dim) {
        query = convolved[query_offset + thread_index];
        key = convolved[key_offset + thread_index];
    }
    query_partial[thread_index] = query * query;
    key_partial[thread_index] = key * key;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            query_partial[thread_index] += query_partial[thread_index + stride];
            key_partial[thread_index] += key_partial[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index < key_head_dim) {
        const float inverse_head_scale = rsqrt(float(key_head_dim));
        const float query_scale = rsqrt(query_partial[0] / float(key_head_dim) + epsilon);
        const float key_scale = rsqrt(key_partial[0] / float(key_head_dim) + epsilon);
        convolved[query_offset + thread_index] = query * query_scale
            * inverse_head_scale * inverse_head_scale;
        convolved[key_offset + thread_index] = key * key_scale * inverse_head_scale;
    }
}

// One threadgroup updates one DeltaNet value head. Each active SIMD lane owns
// one value row and keeps its 128-element reduction private, avoiding 128
// tiny threadgroups per head while preserving the recurrence order.
kernel void qwen38_deltanet_recurrence(
    device const float* convolved [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device const float* a [[buffer(3)]],
    device const float* a_log [[buffer(4)]],
    device const float* dt_bias [[buffer(5)]],
    device float* recurrent [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& key_heads [[buffer(8)]],
    constant uint& value_heads [[buffer(9)]],
    constant uint& key_head_dim [[buffer(10)]],
    constant uint& value_head_dim [[buffer(11)]],
    uint value_head [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint value_index = thread_index;
    if (value_head >= value_heads || value_index >= value_head_dim) {
        return;
    }
    const uint key_elements = key_heads * key_head_dim;
    const uint repeat = value_heads / key_heads;
    const uint key_head = value_head / repeat;
    const uint key_offset = key_elements + key_head * key_head_dim;
    const uint query_offset = key_head * key_head_dim;
    const uint state_base = (value_head * value_head_dim + value_index) * key_head_dim;
    const float decay = exp(-exp(a_log[value_head])
        * qwen38_softplus(a[value_head] + dt_bias[value_head]));
    const float beta = qwen38_sigmoid(b[value_head]);
    float kv_mem = 0.0f;
    for (uint key_index = 0; key_index < key_head_dim; ++key_index) {
        const uint state_index = state_base + key_index;
        const float state_value = recurrent[state_index] * decay;
        recurrent[state_index] = state_value;
        kv_mem += state_value * convolved[key_offset + key_index];
    }
    const uint value_offset = 2 * key_elements + value_head * value_head_dim + value_index;
    const float delta = (convolved[value_offset] - kv_mem) * beta;
    float output_value = 0.0f;
    for (uint key_index = 0; key_index < key_head_dim; ++key_index) {
        const uint state_index = state_base + key_index;
        const float state_value = recurrent[state_index]
            + convolved[key_offset + key_index] * delta;
        recurrent[state_index] = state_value;
        output_value += state_value * convolved[query_offset + key_index];
    }
    output[value_head * value_head_dim + value_index] = output_value;
}

kernel void qwen38_deltanet_gate_norm(
    device float* output [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const float* norm [[buffer(2)]],
    constant uint& value_heads [[buffer(3)]],
    constant uint& value_head_dim [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    threadgroup float* partial [[threadgroup(0)]],
    uint value_head [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    if (value_head >= value_heads) {
        return;
    }
    const uint offset = value_head * value_head_dim;
    const float value = thread_index < value_head_dim ? output[offset + thread_index] : 0.0f;
    partial[thread_index] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            partial[thread_index] += partial[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index < value_head_dim) {
        const float scale = rsqrt(partial[0] / float(value_head_dim) + epsilon);
        output[offset + thread_index] = value * scale * norm[thread_index]
            * qwen38_silu(z[offset + thread_index]);
    }
}

// Layer-major prefill keeps the DeltaNet recurrence causal without returning
// to the CPU for every position. One threadgroup owns a key head and its
// repeated value heads, advancing the complete prompt in order.
kernel void qwen38_deltanet_prefill(
    device const float* qkv [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device const float* a [[buffer(3)]],
    device const float* conv_weight [[buffer(4)]],
    device const float* a_log [[buffer(5)]],
    device const float* dt_bias [[buffer(6)]],
    device const float* norm [[buffer(7)]],
    device float* conv_history [[buffer(8)]],
    device float* recurrent [[buffer(9)]],
    device float* output [[buffer(10)]],
    constant uint& batch_size [[buffer(11)]],
    constant uint& key_heads [[buffer(12)]],
    constant uint& value_heads [[buffer(13)]],
    constant uint& key_head_dim [[buffer(14)]],
    constant uint& value_head_dim [[buffer(15)]],
    constant uint& conv_kernel_size [[buffer(16)]],
    constant float& epsilon [[buffer(17)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint key_head [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    if (key_head >= key_heads) {
        return;
    }

    const uint key_elements = key_heads * key_head_dim;
    const uint value_elements = value_heads * value_head_dim;
    const uint channels = 2 * key_elements + value_elements;
    const uint repeat = value_heads / key_heads;
    const uint history_width = conv_kernel_size - 1;
    threadgroup float* query_values = scratch;
    threadgroup float* key_values = scratch + key_head_dim;
    threadgroup float* query_partial = scratch + 2 * key_head_dim;
    threadgroup float* key_partial = query_partial + 256;
    threadgroup float* state_values = key_partial + 256;

    for (uint token = 0; token < batch_size; ++token) {
        const uint qkv_base = token * channels;
        const uint value_base = token * value_elements;
        if (thread_index < key_head_dim) {
            const uint query_channel = key_head * key_head_dim + thread_index;
            const uint key_channel = key_elements + query_channel;
            const uint query_history_base = query_channel * history_width;
            const uint key_history_base = key_channel * history_width;
            float query_sum = conv_weight[query_channel * conv_kernel_size + history_width]
                * qkv[qkv_base + query_channel];
            float key_sum = conv_weight[key_channel * conv_kernel_size + history_width]
                * qkv[qkv_base + key_channel];
            for (uint tap = 0; tap < history_width; ++tap) {
                query_sum += conv_weight[query_channel * conv_kernel_size + tap]
                    * conv_history[query_history_base + tap];
                key_sum += conv_weight[key_channel * conv_kernel_size + tap]
                    * conv_history[key_history_base + tap];
            }
            query_values[thread_index] = qwen38_silu(query_sum);
            key_values[thread_index] = qwen38_silu(key_sum);
            if (history_width > 0) {
                for (uint tap = 0; tap + 1 < history_width; ++tap) {
                    conv_history[query_history_base + tap] = conv_history[query_history_base + tap + 1];
                    conv_history[key_history_base + tap] = conv_history[key_history_base + tap + 1];
                }
                conv_history[query_history_base + history_width - 1] = qkv[qkv_base + query_channel];
                conv_history[key_history_base + history_width - 1] = qkv[qkv_base + key_channel];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        query_partial[thread_index] = thread_index < key_head_dim
            ? query_values[thread_index] * query_values[thread_index]
            : 0.0f;
        key_partial[thread_index] = thread_index < key_head_dim
            ? key_values[thread_index] * key_values[thread_index]
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (thread_index < stride) {
                query_partial[thread_index] += query_partial[thread_index + stride];
                key_partial[thread_index] += key_partial[thread_index + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_index < key_head_dim) {
            const float inverse_head_scale = rsqrt(float(key_head_dim));
            const float query_scale = rsqrt(query_partial[0] / float(key_head_dim) + epsilon);
            const float key_scale = rsqrt(key_partial[0] / float(key_head_dim) + epsilon);
            query_values[thread_index] = query_values[thread_index] * query_scale
                * inverse_head_scale * inverse_head_scale;
            key_values[thread_index] = key_values[thread_index] * key_scale * inverse_head_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint local_value = thread_index;
             local_value < repeat * value_head_dim;
             local_value += 256) {
            const uint value_head = key_head * repeat + local_value / value_head_dim;
            const uint value_index = local_value % value_head_dim;
            const uint value_channel = 2 * key_elements + value_head * value_head_dim + value_index;
            const uint value_history_base = value_channel * history_width;
            float value_sum = conv_weight[value_channel * conv_kernel_size + history_width]
                * qkv[qkv_base + value_channel];
            for (uint tap = 0; tap < history_width; ++tap) {
                value_sum += conv_weight[value_channel * conv_kernel_size + tap]
                    * conv_history[value_history_base + tap];
            }
            if (history_width > 0) {
                for (uint tap = 0; tap + 1 < history_width; ++tap) {
                    conv_history[value_history_base + tap] = conv_history[value_history_base + tap + 1];
                }
                conv_history[value_history_base + history_width - 1] = qkv[qkv_base + value_channel];
            }
            const float decay = exp(-exp(a_log[value_head])
                * qwen38_softplus(a[token * value_heads + value_head] + dt_bias[value_head]));
            const float beta = qwen38_sigmoid(b[token * value_heads + value_head]);
            const uint state_base = (value_head * value_head_dim + value_index) * key_head_dim;
            float kv_mem = 0.0f;
            for (uint key_index = 0; key_index < key_head_dim; ++key_index) {
                const uint state_index = state_base + key_index;
                const float state_value = recurrent[state_index] * decay;
                recurrent[state_index] = state_value;
                kv_mem += state_value * key_values[key_index];
            }
            const float delta = (qwen38_silu(value_sum) - kv_mem) * beta;
            float output_value = 0.0f;
            for (uint key_index = 0; key_index < key_head_dim; ++key_index) {
                const uint state_index = state_base + key_index;
                const float state_value = recurrent[state_index] + key_values[key_index] * delta;
                recurrent[state_index] = state_value;
                output_value += state_value * query_values[key_index];
            }
            output[value_base + value_head * value_head_dim + value_index] = output_value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint local_head = 0; local_head < repeat; ++local_head) {
            const uint value_head = key_head * repeat + local_head;
            const uint output_offset = value_base + value_head * value_head_dim;
            state_values[thread_index] = thread_index < value_head_dim
                ? output[output_offset + thread_index] * output[output_offset + thread_index]
                : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 128; stride > 0; stride >>= 1) {
                if (thread_index < stride) {
                    state_values[thread_index] += state_values[thread_index + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (thread_index < value_head_dim) {
                const float scale = rsqrt(state_values[0] / float(value_head_dim) + epsilon);
                const uint output_index = output_offset + thread_index;
                output[output_index] = output[output_index] * scale * norm[thread_index]
                    * qwen38_silu(z[output_index]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Full-attention KV is stored as Q8 plus a scale per token/head. Pages are
// grown by the Rust runtime only when the active sequence crosses capacity.
kernel void qwen38_q8_kv_append(
    device const float* key_input [[buffer(0)]],
    device const float* value_input [[buffer(1)]],
    device char* keys [[buffer(2)]],
    device char* values [[buffer(3)]],
    device float* key_scales [[buffer(4)]],
    device float* value_scales [[buffer(5)]],
    constant uint& kv_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& token_index [[buffer(8)]],
    threadgroup float* partial [[threadgroup(0)]],
    uint2 head_kind [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint head = head_kind.x;
    const uint kind = head_kind.y;
    if (head >= kv_heads || kind > 1) {
        return;
    }
    const uint input_offset = head * head_dim;
    const float value = thread_index < head_dim
        ? (kind == 0 ? key_input[input_offset + thread_index]
                     : value_input[input_offset + thread_index])
        : 0.0f;
    partial[thread_index] = abs(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            partial[thread_index] = max(partial[thread_index], partial[thread_index + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = max(partial[0] / 127.0f, FLT_MIN);
    const uint scale_index = token_index * kv_heads + head;
    const uint output_offset = (token_index * kv_heads + head) * head_dim;
    if (thread_index == 0) {
        if (kind == 0) {
            key_scales[scale_index] = scale;
        } else {
            value_scales[scale_index] = scale;
        }
    }
    if (thread_index < head_dim) {
        const char quantized = char(round(clamp(value / scale, -127.0f, 127.0f)));
        if (kind == 0) {
            keys[output_offset + thread_index] = quantized;
        } else {
            values[output_offset + thread_index] = quantized;
        }
    }
}

// Quantize an entire prompt's K/V rows before causal attention reads them.
// The scalar group index avoids a multi-dimensional threadgroup ABI and maps
// to (prompt row, KV head, key-or-value) deterministically.
kernel void qwen38_q8_kv_append_prefill(
    device const float* key_input [[buffer(0)]],
    device const float* value_input [[buffer(1)]],
    device char* keys [[buffer(2)]],
    device char* values [[buffer(3)]],
    device float* key_scales [[buffer(4)]],
    device float* value_scales [[buffer(5)]],
    constant uint& kv_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& start_token [[buffer(8)]],
    constant uint& batch_size [[buffer(9)]],
    threadgroup float* partial [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint groups_per_token = kv_heads * 2;
    const uint prompt_token = group / groups_per_token;
    if (prompt_token >= batch_size) {
        return;
    }
    const uint within_token = group % groups_per_token;
    const uint head = within_token % kv_heads;
    const uint kind = within_token / kv_heads;
    const uint input_offset = (prompt_token * kv_heads + head) * head_dim;
    const float value = thread_index < head_dim
        ? (kind == 0 ? key_input[input_offset + thread_index]
                     : value_input[input_offset + thread_index])
        : 0.0f;
    partial[thread_index] = abs(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            partial[thread_index] = max(partial[thread_index], partial[thread_index + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = max(partial[0] / 127.0f, FLT_MIN);
    const uint token = start_token + prompt_token;
    const uint scale_index = token * kv_heads + head;
    const uint output_offset = (token * kv_heads + head) * head_dim;
    if (thread_index == 0) {
        if (kind == 0) {
            key_scales[scale_index] = scale;
        } else {
            value_scales[scale_index] = scale;
        }
    }
    if (thread_index < head_dim) {
        const char quantized = char(round(clamp(value / scale, -127.0f, 127.0f)));
        if (kind == 0) {
            keys[output_offset + thread_index] = quantized;
        } else {
            values[output_offset + thread_index] = quantized;
        }
    }
}

// Stable online softmax keeps causal prefill memory linear in prompt length.
// Each threadgroup owns one prompt query/head pair and streams its visible KV
// rows directly from the Q8 cache, avoiding a T^2 score matrix.
kernel void qwen38_gqa_q8_prefill_attention(
    device const float* query [[buffer(0)]],
    device const char* keys [[buffer(1)]],
    device const float* key_scales [[buffer(2)]],
    device const char* values [[buffer(3)]],
    device const float* value_scales [[buffer(4)]],
    device const float* gate [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& start_token [[buffer(7)]],
    constant uint& total_length [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& kv_heads [[buffer(10)]],
    constant uint& head_dim [[buffer(11)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint query_index = group / num_heads;
    const uint head = group % num_heads;
    const uint kv_head = head * kv_heads / num_heads;
    const uint query_offset = (query_index * num_heads + head) * head_dim;
    const uint visible_tokens = min(start_token + query_index + 1, total_length);
    threadgroup float* partial = scratch;
    threadgroup float* accumulator = scratch + 256;
    threadgroup float* state = accumulator + 256;
    if (thread_index < head_dim) {
        accumulator[thread_index] = 0.0f;
    }
    if (thread_index == 0) {
        state[0] = -INFINITY;
        state[1] = 0.0f;
        state[2] = 0.0f;
        state[3] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint token = 0; token < visible_tokens; ++token) {
        const uint key_offset = (token * kv_heads + kv_head) * head_dim;
        partial[thread_index] = thread_index < head_dim
            ? query[query_offset + thread_index] * float(keys[key_offset + thread_index])
                * key_scales[token * kv_heads + kv_head]
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (thread_index < stride) {
                partial[thread_index] += partial[thread_index + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_index == 0) {
            const float score = partial[0] * rsqrt(float(head_dim));
            const float maximum = max(state[0], score);
            const float previous_scale = exp(state[0] - maximum);
            const float token_scale = exp(score - maximum);
            state[0] = maximum;
            state[1] = state[1] * previous_scale + token_scale;
            state[2] = previous_scale;
            state[3] = token_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index < head_dim) {
            const uint value_offset = key_offset + thread_index;
            accumulator[thread_index] = accumulator[thread_index] * state[2]
                + state[3] * float(values[value_offset])
                    * value_scales[token * kv_heads + kv_head];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index < head_dim) {
        const uint output_offset = query_offset + thread_index;
        output[output_offset] = accumulator[thread_index] / max(state[1], FLT_MIN)
            * qwen38_sigmoid(gate[output_offset]);
    }
}

kernel void qwen38_gqa_q8_scores(
    device const float* query [[buffer(0)]],
    device const char* keys [[buffer(1)]],
    device const float* key_scales [[buffer(2)]],
    device float* scores [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& kv_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    threadgroup float* partial [[threadgroup(0)]],
    uint head [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    if (head >= num_heads) {
        return;
    }
    const uint kv_head = head * kv_heads / num_heads;
    const uint query_offset = head * head_dim;
    const uint score_offset = head * sequence_length;
    float local_max = -INFINITY;
    for (uint token = thread_index; token < sequence_length; token += 256) {
        const uint key_offset = (token * kv_heads + kv_head) * head_dim;
        const float scale = key_scales[token * kv_heads + kv_head];
        float score = 0.0f;
        for (uint dimension = 0; dimension < head_dim; ++dimension) {
            score += query[query_offset + dimension] * float(keys[key_offset + dimension]) * scale;
        }
        score *= rsqrt(float(head_dim));
        scores[score_offset + token] = score;
        local_max = max(local_max, score);
    }
    partial[thread_index] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            partial[thread_index] = max(partial[thread_index], partial[thread_index + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float maximum = partial[0];
    float local_sum = 0.0f;
    for (uint token = thread_index; token < sequence_length; token += 256) {
        const float probability = exp(scores[score_offset + token] - maximum);
        scores[score_offset + token] = probability;
        local_sum += probability;
    }
    partial[thread_index] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            partial[thread_index] += partial[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint token = thread_index; token < sequence_length; token += 256) {
        scores[score_offset + token] /= max(partial[0], FLT_MIN);
    }
}

kernel void qwen38_gqa_q8_values(
    device const float* scores [[buffer(0)]],
    device const char* values [[buffer(1)]],
    device const float* value_scales [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& sequence_length [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint dimension [[thread_index_in_threadgroup]]) {
    if (head >= num_heads || dimension >= head_dim) {
        return;
    }
    const uint kv_head = head * kv_heads / num_heads;
    const uint score_offset = head * sequence_length;
    float sum = 0.0f;
    for (uint token = 0; token < sequence_length; ++token) {
        const uint value_offset = (token * kv_heads + kv_head) * head_dim + dimension;
        sum += scores[score_offset + token] * float(values[value_offset])
            * value_scales[token * kv_heads + kv_head];
    }
    const uint output_offset = head * head_dim + dimension;
    output[output_offset] = sum * qwen38_sigmoid(gate[output_offset]);
}
