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

inline float qwen38_sigmoid(float value) {
    return 1.0f / (1.0f + exp(-value));
}

inline float qwen38_silu(float value) {
    return value * qwen38_sigmoid(value);
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
