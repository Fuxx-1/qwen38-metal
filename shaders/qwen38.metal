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
