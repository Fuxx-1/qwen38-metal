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
