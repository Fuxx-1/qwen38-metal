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

inline ushort qwen38_float_to_bf16(float value) {
    const uint bits = as_type<uint>(value);
    const uint rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
    return ushort(rounded >> 16u);
}

inline float qwen38_q4_word_dot_values(
    uint packed,
    float4 values0,
    float4 values1) {
    const float4 quantized0 = float4(
        float(packed & 0xFu),
        float((packed >> 4u) & 0xFu),
        float((packed >> 8u) & 0xFu),
        float((packed >> 12u) & 0xFu));
    const float4 quantized1 = float4(
        float((packed >> 16u) & 0xFu),
        float((packed >> 20u) & 0xFu),
        float((packed >> 24u) & 0xFu),
        float((packed >> 28u) & 0xFu));
    return dot(values0, quantized0) + dot(values1, quantized1);
}

inline float qwen38_q4_word_dot(
    uint packed,
    device const float* input,
    uint input_offset) {
    const float4 values0 = float4(
        input[input_offset],
        input[input_offset + 1u],
        input[input_offset + 2u],
        input[input_offset + 3u]);
    const float4 values1 = float4(
        input[input_offset + 4u],
        input[input_offset + 5u],
        input[input_offset + 6u],
        input[input_offset + 7u]);
    return qwen38_q4_word_dot_values(packed, values0, values1);
}

inline float qwen38_q4_word_dot_threadgroup(
    uint packed,
    threadgroup const float* input,
    uint input_offset) {
    const float4 values0 = float4(
        input[input_offset],
        input[input_offset + 1u],
        input[input_offset + 2u],
        input[input_offset + 3u]);
    const float4 values1 = float4(
        input[input_offset + 4u],
        input[input_offset + 5u],
        input[input_offset + 6u],
        input[input_offset + 7u]);
    const float4 quantized0 = float4(
        float(packed & 0xFu),
        float((packed >> 4u) & 0xFu),
        float((packed >> 8u) & 0xFu),
        float((packed >> 12u) & 0xFu));
    const float4 quantized1 = float4(
        float((packed >> 16u) & 0xFu),
        float((packed >> 20u) & 0xFu),
        float((packed >> 24u) & 0xFu),
        float((packed >> 28u) & 0xFu));
    return dot(values0, quantized0) + dot(values1, quantized1);
}

// Decode-oriented path: one SIMD group owns one output row. The affine
// parameters are reused for all eight packed words in a 64-element group,
// and simd_sum removes the 256-thread shared-memory reduction from the hot
// single-token path.
kernel void qwen38_q4_affine_matvec_simd(
    device const float* input [[buffer(0)]],
    device const uint* packed_weights [[buffer(1)]],
    device const ushort* scales [[buffer(2)]],
    device const ushort* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& words_per_row [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint groups_per_row = words_per_row / 8u;
    float sum = 0.0f;
    for (uint group = lane; group < groups_per_row; group += 32u) {
        const float scale = qwen38_bf16_to_float(scales[row * groups_per_row + group]);
        const float bias = qwen38_bf16_to_float(biases[row * groups_per_row + group]);
        const uint word_base = row * words_per_row + group * 8u;
        const uint input_base = group * 64u;
        float dot_sum = 0.0f;
        float input_sum = 0.0f;
        for (uint word = 0; word < 8u; ++word) {
            const uint packed = packed_weights[word_base + word];
            const uint offset = input_base + word * 8u;
            dot_sum += qwen38_q4_word_dot(packed, input, offset);
            input_sum += input[offset] + input[offset + 1u]
                + input[offset + 2u] + input[offset + 3u]
                + input[offset + 4u] + input[offset + 5u]
                + input[offset + 6u] + input[offset + 7u];
        }
        sum += dot_sum * scale + input_sum * bias;
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        output[row] = sum;
    }
}

kernel void qwen38_q4_affine_matvec_simd_unaligned(
    device const float* input [[buffer(0)]],
    device const uchar* packed_weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const uchar* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& words_per_row [[buffer(5)]],
    constant ulong& weight_byte_offset [[buffer(6)]],
    constant ulong& scale_byte_offset [[buffer(7)]],
    constant ulong& bias_byte_offset [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint groups_per_row = words_per_row / 8u;
    float sum = 0.0f;
    for (uint group = lane; group < groups_per_row; group += 32u) {
        const ulong parameter_offset = ulong(row * groups_per_row + group) * 2ul;
        const ushort scale_bits = ushort(scales[scale_byte_offset + parameter_offset])
            | (ushort(scales[scale_byte_offset + parameter_offset + 1ul]) << 8u);
        const ushort bias_bits = ushort(biases[bias_byte_offset + parameter_offset])
            | (ushort(biases[bias_byte_offset + parameter_offset + 1ul]) << 8u);
        const float scale = qwen38_bf16_to_float(scale_bits);
        const float bias = qwen38_bf16_to_float(bias_bits);
        const ulong word_base = weight_byte_offset
            + ulong(row * words_per_row + group * 8u) * 4ul;
        const uint input_base = group * 64u;
        float dot_sum = 0.0f;
        float input_sum = 0.0f;
        for (uint word = 0; word < 8u; ++word) {
            const ulong address = word_base + ulong(word) * 4ul;
            const uint packed = uint(packed_weights[address])
                | (uint(packed_weights[address + 1ul]) << 8u)
                | (uint(packed_weights[address + 2ul]) << 16u)
                | (uint(packed_weights[address + 3ul]) << 24u);
            const uint offset = input_base + word * 8u;
            dot_sum += qwen38_q4_word_dot(packed, input, offset);
            input_sum += input[offset] + input[offset + 1u]
                + input[offset + 2u] + input[offset + 3u]
                + input[offset + 4u] + input[offset + 5u]
                + input[offset + 6u] + input[offset + 7u];
        }
        sum += dot_sum * scale + input_sum * bias;
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        output[row] = sum;
    }
}

// Mirrors MLX's affine qmv_fast scheduling for the Qwen Q4 layout. A 64-thread
// threadgroup contains two SIMD groups; each SIMD group owns four output rows
// and streams one 16-value activation fragment through all four rows. This
// reuses activation loads without staging the full vector in threadgroup memory.
kernel void qwen38_q4_affine_matvec_mlx_fast(
    device const float* input [[buffer(0)]],
    device const uint* packed_weights [[buffer(1)]],
    device const ushort* scales [[buffer(2)]],
    device const ushort* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& words_per_row [[buffer(5)]],
    constant uint& output_rows [[buffer(6)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint values_per_lane = 16u;
    constexpr uint values_per_block = values_per_lane * 32u;
    constexpr uint rows_per_simd = 4u;
    const uint groups_per_row = words_per_row / 8u;
    const uint input_elements = words_per_row * 8u;
    const uint row0 = tile.x * 8u + simd_group * rows_per_simd;
    float result0 = 0.0f;
    float result1 = 0.0f;
    float result2 = 0.0f;
    float result3 = 0.0f;

    for (uint input_base = lane * values_per_lane;
         input_base < input_elements;
         input_base += values_per_block) {
        const float4 values0 = float4(
            input[input_base], input[input_base + 1u],
            input[input_base + 2u], input[input_base + 3u]);
        const float4 values1 = float4(
            input[input_base + 4u], input[input_base + 5u],
            input[input_base + 6u], input[input_base + 7u]);
        const float4 values2 = float4(
            input[input_base + 8u], input[input_base + 9u],
            input[input_base + 10u], input[input_base + 11u]);
        const float4 values3 = float4(
            input[input_base + 12u], input[input_base + 13u],
            input[input_base + 14u], input[input_base + 15u]);
        const float input_sum = values0.x + values0.y + values0.z + values0.w
            + values1.x + values1.y + values1.z + values1.w
            + values2.x + values2.y + values2.z + values2.w
            + values3.x + values3.y + values3.z + values3.w;
        const uint group = input_base / 64u;
        const uint word = input_base / 8u;

        if (row0 < output_rows) {
            const uint weight_base = row0 * words_per_row + word;
            const float scale = qwen38_bf16_to_float(scales[row0 * groups_per_row + group]);
            const float bias = qwen38_bf16_to_float(biases[row0 * groups_per_row + group]);
            const float dot_sum = qwen38_q4_word_dot_values(
                    packed_weights[weight_base], values0, values1)
                + qwen38_q4_word_dot_values(packed_weights[weight_base + 1u], values2, values3);
            result0 += dot_sum * scale + input_sum * bias;
        }
        if (row0 + 1u < output_rows) {
            const uint row = row0 + 1u;
            const uint weight_base = row * words_per_row + word;
            const float scale = qwen38_bf16_to_float(scales[row * groups_per_row + group]);
            const float bias = qwen38_bf16_to_float(biases[row * groups_per_row + group]);
            const float dot_sum = qwen38_q4_word_dot_values(
                    packed_weights[weight_base], values0, values1)
                + qwen38_q4_word_dot_values(packed_weights[weight_base + 1u], values2, values3);
            result1 += dot_sum * scale + input_sum * bias;
        }
        if (row0 + 2u < output_rows) {
            const uint row = row0 + 2u;
            const uint weight_base = row * words_per_row + word;
            const float scale = qwen38_bf16_to_float(scales[row * groups_per_row + group]);
            const float bias = qwen38_bf16_to_float(biases[row * groups_per_row + group]);
            const float dot_sum = qwen38_q4_word_dot_values(
                    packed_weights[weight_base], values0, values1)
                + qwen38_q4_word_dot_values(packed_weights[weight_base + 1u], values2, values3);
            result2 += dot_sum * scale + input_sum * bias;
        }
        if (row0 + 3u < output_rows) {
            const uint row = row0 + 3u;
            const uint weight_base = row * words_per_row + word;
            const float scale = qwen38_bf16_to_float(scales[row * groups_per_row + group]);
            const float bias = qwen38_bf16_to_float(biases[row * groups_per_row + group]);
            const float dot_sum = qwen38_q4_word_dot_values(
                    packed_weights[weight_base], values0, values1)
                + qwen38_q4_word_dot_values(packed_weights[weight_base + 1u], values2, values3);
            result3 += dot_sum * scale + input_sum * bias;
        }
    }

    result0 = simd_sum(result0);
    result1 = simd_sum(result1);
    result2 = simd_sum(result2);
    result3 = simd_sum(result3);
    if (lane == 0u) {
        if (row0 < output_rows) {
            output[row0] = result0;
        }
        if (row0 + 1u < output_rows) {
            output[row0 + 1u] = result1;
        }
        if (row0 + 2u < output_rows) {
            output[row0 + 2u] = result2;
        }
        if (row0 + 3u < output_rows) {
            output[row0 + 3u] = result3;
        }
    }
}

kernel void qwen38_q4_affine_matvec_mlx_fast_unaligned(
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
    uint3 tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint values_per_lane = 16u;
    constexpr uint values_per_block = values_per_lane * 32u;
    constexpr uint rows_per_simd = 4u;
    const uint groups_per_row = words_per_row / 8u;
    const uint input_elements = words_per_row * 8u;
    const uint row0 = tile.x * 8u + simd_group * rows_per_simd;
    float result0 = 0.0f;
    float result1 = 0.0f;
    float result2 = 0.0f;
    float result3 = 0.0f;

    for (uint input_base = lane * values_per_lane;
         input_base < input_elements;
         input_base += values_per_block) {
        const float4 values0 = float4(
            input[input_base], input[input_base + 1u],
            input[input_base + 2u], input[input_base + 3u]);
        const float4 values1 = float4(
            input[input_base + 4u], input[input_base + 5u],
            input[input_base + 6u], input[input_base + 7u]);
        const float4 values2 = float4(
            input[input_base + 8u], input[input_base + 9u],
            input[input_base + 10u], input[input_base + 11u]);
        const float4 values3 = float4(
            input[input_base + 12u], input[input_base + 13u],
            input[input_base + 14u], input[input_base + 15u]);
        const float input_sum = values0.x + values0.y + values0.z + values0.w
            + values1.x + values1.y + values1.z + values1.w
            + values2.x + values2.y + values2.z + values2.w
            + values3.x + values3.y + values3.z + values3.w;
        const uint group = input_base / 64u;
        const uint word = input_base / 8u;

        if (row0 < output_rows) {
            const ulong parameter = ulong(row0 * groups_per_row + group) * 2ul;
            const ulong weight_base = weight_byte_offset + ulong(row0 * words_per_row + word) * 4ul;
            const float scale = qwen38_bf16_to_float(
                ushort(scales[scale_byte_offset + parameter])
                    | (ushort(scales[scale_byte_offset + parameter + 1ul]) << 8u));
            const float bias = qwen38_bf16_to_float(
                ushort(biases[bias_byte_offset + parameter])
                    | (ushort(biases[bias_byte_offset + parameter + 1ul]) << 8u));
            const uint packed0 = uint(packed_weights[weight_base])
                | (uint(packed_weights[weight_base + 1ul]) << 8u)
                | (uint(packed_weights[weight_base + 2ul]) << 16u)
                | (uint(packed_weights[weight_base + 3ul]) << 24u);
            const uint packed1 = uint(packed_weights[weight_base + 4ul])
                | (uint(packed_weights[weight_base + 5ul]) << 8u)
                | (uint(packed_weights[weight_base + 6ul]) << 16u)
                | (uint(packed_weights[weight_base + 7ul]) << 24u);
            result0 += (qwen38_q4_word_dot_values(packed0, values0, values1)
                    + qwen38_q4_word_dot_values(packed1, values2, values3)) * scale
                + input_sum * bias;
        }
        if (row0 + 1u < output_rows) {
            const uint row = row0 + 1u;
            const ulong parameter = ulong(row * groups_per_row + group) * 2ul;
            const ulong weight_base = weight_byte_offset + ulong(row * words_per_row + word) * 4ul;
            const float scale = qwen38_bf16_to_float(
                ushort(scales[scale_byte_offset + parameter])
                    | (ushort(scales[scale_byte_offset + parameter + 1ul]) << 8u));
            const float bias = qwen38_bf16_to_float(
                ushort(biases[bias_byte_offset + parameter])
                    | (ushort(biases[bias_byte_offset + parameter + 1ul]) << 8u));
            const uint packed0 = uint(packed_weights[weight_base])
                | (uint(packed_weights[weight_base + 1ul]) << 8u)
                | (uint(packed_weights[weight_base + 2ul]) << 16u)
                | (uint(packed_weights[weight_base + 3ul]) << 24u);
            const uint packed1 = uint(packed_weights[weight_base + 4ul])
                | (uint(packed_weights[weight_base + 5ul]) << 8u)
                | (uint(packed_weights[weight_base + 6ul]) << 16u)
                | (uint(packed_weights[weight_base + 7ul]) << 24u);
            result1 += (qwen38_q4_word_dot_values(packed0, values0, values1)
                    + qwen38_q4_word_dot_values(packed1, values2, values3)) * scale
                + input_sum * bias;
        }
        if (row0 + 2u < output_rows) {
            const uint row = row0 + 2u;
            const ulong parameter = ulong(row * groups_per_row + group) * 2ul;
            const ulong weight_base = weight_byte_offset + ulong(row * words_per_row + word) * 4ul;
            const float scale = qwen38_bf16_to_float(
                ushort(scales[scale_byte_offset + parameter])
                    | (ushort(scales[scale_byte_offset + parameter + 1ul]) << 8u));
            const float bias = qwen38_bf16_to_float(
                ushort(biases[bias_byte_offset + parameter])
                    | (ushort(biases[bias_byte_offset + parameter + 1ul]) << 8u));
            const uint packed0 = uint(packed_weights[weight_base])
                | (uint(packed_weights[weight_base + 1ul]) << 8u)
                | (uint(packed_weights[weight_base + 2ul]) << 16u)
                | (uint(packed_weights[weight_base + 3ul]) << 24u);
            const uint packed1 = uint(packed_weights[weight_base + 4ul])
                | (uint(packed_weights[weight_base + 5ul]) << 8u)
                | (uint(packed_weights[weight_base + 6ul]) << 16u)
                | (uint(packed_weights[weight_base + 7ul]) << 24u);
            result2 += (qwen38_q4_word_dot_values(packed0, values0, values1)
                    + qwen38_q4_word_dot_values(packed1, values2, values3)) * scale
                + input_sum * bias;
        }
        if (row0 + 3u < output_rows) {
            const uint row = row0 + 3u;
            const ulong parameter = ulong(row * groups_per_row + group) * 2ul;
            const ulong weight_base = weight_byte_offset + ulong(row * words_per_row + word) * 4ul;
            const float scale = qwen38_bf16_to_float(
                ushort(scales[scale_byte_offset + parameter])
                    | (ushort(scales[scale_byte_offset + parameter + 1ul]) << 8u));
            const float bias = qwen38_bf16_to_float(
                ushort(biases[bias_byte_offset + parameter])
                    | (ushort(biases[bias_byte_offset + parameter + 1ul]) << 8u));
            const uint packed0 = uint(packed_weights[weight_base])
                | (uint(packed_weights[weight_base + 1ul]) << 8u)
                | (uint(packed_weights[weight_base + 2ul]) << 16u)
                | (uint(packed_weights[weight_base + 3ul]) << 24u);
            const uint packed1 = uint(packed_weights[weight_base + 4ul])
                | (uint(packed_weights[weight_base + 5ul]) << 8u)
                | (uint(packed_weights[weight_base + 6ul]) << 16u)
                | (uint(packed_weights[weight_base + 7ul]) << 24u);
            result3 += (qwen38_q4_word_dot_values(packed0, values0, values1)
                    + qwen38_q4_word_dot_values(packed1, values2, values3)) * scale
                + input_sum * bias;
        }
    }

    result0 = simd_sum(result0);
    result1 = simd_sum(result1);
    result2 = simd_sum(result2);
    result3 = simd_sum(result3);
    if (lane == 0u) {
        if (row0 < output_rows) {
            output[row0] = result0;
        }
        if (row0 + 1u < output_rows) {
            output[row0 + 1u] = result1;
        }
        if (row0 + 2u < output_rows) {
            output[row0 + 2u] = result2;
        }
        if (row0 + 3u < output_rows) {
            output[row0 + 3u] = result3;
        }
    }
}

// Decode is bandwidth-bound. Eight output rows share one staged activation
// vector, avoiding the global activation rereads made by one-row SIMD groups.
// Qwen's hidden and attention-output widths fit within the Apple GPU's
// threadgroup-memory limit, while the wider MLP-down input uses the tiled
// variant below.
#define QWEN38_DECODE_SHARED_ROWS 8u
#define QWEN38_DECODE_SHARED_THREADS 256u

kernel void qwen38_q4_affine_matvec_shared(
    device const float* input [[buffer(0)]],
    device const uint* packed_weights [[buffer(1)]],
    device const ushort* scales [[buffer(2)]],
    device const ushort* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& words_per_row [[buffer(5)]],
    constant uint& output_rows [[buffer(6)]],
    threadgroup float* input_tile [[threadgroup(0)]],
    uint output_group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint input_width = words_per_row * 8u;
    for (uint index = thread_index; index < input_width;
         index += QWEN38_DECODE_SHARED_THREADS) {
        input_tile[index] = input[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint local_row = thread_index / 32u;
    const uint lane = thread_index % 32u;
    const uint row = output_group * QWEN38_DECODE_SHARED_ROWS + local_row;
    const uint groups_per_row = words_per_row / 8u;
    float sum = 0.0f;
    if (row < output_rows) {
        for (uint group = lane; group < groups_per_row; group += 32u) {
            const float scale = qwen38_bf16_to_float(scales[row * groups_per_row + group]);
            const float bias = qwen38_bf16_to_float(biases[row * groups_per_row + group]);
            const uint word_base = row * words_per_row + group * 8u;
            const uint input_base = group * 64u;
            float dot_sum = 0.0f;
            float input_sum = 0.0f;
            for (uint word = 0; word < 8u; ++word) {
                const uint offset = input_base + word * 8u;
                dot_sum += qwen38_q4_word_dot_threadgroup(
                    packed_weights[word_base + word], input_tile, offset);
                input_sum += input_tile[offset] + input_tile[offset + 1u]
                    + input_tile[offset + 2u] + input_tile[offset + 3u]
                    + input_tile[offset + 4u] + input_tile[offset + 5u]
                    + input_tile[offset + 6u] + input_tile[offset + 7u];
            }
            sum += dot_sum * scale + input_sum * bias;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0u && row < output_rows) {
        output[row] = sum;
    }
}

kernel void qwen38_q4_affine_matvec_shared_unaligned(
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
    threadgroup float* input_tile [[threadgroup(0)]],
    uint output_group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint input_width = words_per_row * 8u;
    for (uint index = thread_index; index < input_width;
         index += QWEN38_DECODE_SHARED_THREADS) {
        input_tile[index] = input[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint local_row = thread_index / 32u;
    const uint lane = thread_index % 32u;
    const uint row = output_group * QWEN38_DECODE_SHARED_ROWS + local_row;
    const uint groups_per_row = words_per_row / 8u;
    float sum = 0.0f;
    if (row < output_rows) {
        for (uint group = lane; group < groups_per_row; group += 32u) {
            const ulong parameter_offset = ulong(row * groups_per_row + group) * 2ul;
            const ulong scale_address = scale_byte_offset + parameter_offset;
            const ulong bias_address = bias_byte_offset + parameter_offset;
            const ushort scale_bits = ushort(scales[scale_address])
                | (ushort(scales[scale_address + 1ul]) << 8u);
            const ushort bias_bits = ushort(biases[bias_address])
                | (ushort(biases[bias_address + 1ul]) << 8u);
            const float scale = qwen38_bf16_to_float(scale_bits);
            const float bias = qwen38_bf16_to_float(bias_bits);
            const ulong word_base = weight_byte_offset
                + ulong(row * words_per_row + group * 8u) * 4ul;
            const uint input_base = group * 64u;
            float dot_sum = 0.0f;
            float input_sum = 0.0f;
            for (uint word = 0; word < 8u; ++word) {
                const ulong address = word_base + ulong(word) * 4ul;
                const uint packed = uint(packed_weights[address])
                    | (uint(packed_weights[address + 1ul]) << 8u)
                    | (uint(packed_weights[address + 2ul]) << 16u)
                    | (uint(packed_weights[address + 3ul]) << 24u);
                const uint offset = input_base + word * 8u;
                dot_sum += qwen38_q4_word_dot_threadgroup(packed, input_tile, offset);
                input_sum += input_tile[offset] + input_tile[offset + 1u]
                    + input_tile[offset + 2u] + input_tile[offset + 3u]
                    + input_tile[offset + 4u] + input_tile[offset + 5u]
                    + input_tile[offset + 6u] + input_tile[offset + 7u];
            }
            sum += dot_sum * scale + input_sum * bias;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0u && row < output_rows) {
        output[row] = sum;
    }
}

#define QWEN38_DECODE_TILED_INPUTS 2048u

kernel void qwen38_q4_affine_matvec_tiled(
    device const float* input [[buffer(0)]],
    device const uint* packed_weights [[buffer(1)]],
    device const ushort* scales [[buffer(2)]],
    device const ushort* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& words_per_row [[buffer(5)]],
    constant uint& output_rows [[buffer(6)]],
    threadgroup float* input_tile [[threadgroup(0)]],
    uint output_group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint input_width = words_per_row * 8u;
    const uint local_row = thread_index / 32u;
    const uint lane = thread_index % 32u;
    const uint row = output_group * QWEN38_DECODE_SHARED_ROWS + local_row;
    const uint groups_per_row = words_per_row / 8u;
    float sum = 0.0f;

    for (uint input_base = 0u; input_base < input_width;
         input_base += QWEN38_DECODE_TILED_INPUTS) {
        const uint tile_elements = min(QWEN38_DECODE_TILED_INPUTS, input_width - input_base);
        for (uint index = thread_index; index < tile_elements;
             index += QWEN38_DECODE_SHARED_THREADS) {
            input_tile[index] = input[input_base + index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < output_rows) {
            const uint first_group = input_base / 64u;
            const uint groups_in_tile = tile_elements / 64u;
            for (uint group = first_group + lane;
                 group < first_group + groups_in_tile;
                 group += 32u) {
                const float scale = qwen38_bf16_to_float(scales[row * groups_per_row + group]);
                const float bias = qwen38_bf16_to_float(biases[row * groups_per_row + group]);
                const uint word_base = row * words_per_row + group * 8u;
                const uint tile_offset = (group - first_group) * 64u;
                float dot_sum = 0.0f;
                float input_sum = 0.0f;
                for (uint word = 0; word < 8u; ++word) {
                    const uint offset = tile_offset + word * 8u;
                    dot_sum += qwen38_q4_word_dot_threadgroup(
                        packed_weights[word_base + word], input_tile, offset);
                    input_sum += input_tile[offset] + input_tile[offset + 1u]
                        + input_tile[offset + 2u] + input_tile[offset + 3u]
                        + input_tile[offset + 4u] + input_tile[offset + 5u]
                        + input_tile[offset + 6u] + input_tile[offset + 7u];
                }
                sum += dot_sum * scale + input_sum * bias;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum = simd_sum(sum);
    if (lane == 0u && row < output_rows) {
        output[row] = sum;
    }
}

kernel void qwen38_q4_affine_matvec_tiled_unaligned(
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
    threadgroup float* input_tile [[threadgroup(0)]],
    uint output_group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint input_width = words_per_row * 8u;
    const uint local_row = thread_index / 32u;
    const uint lane = thread_index % 32u;
    const uint row = output_group * QWEN38_DECODE_SHARED_ROWS + local_row;
    const uint groups_per_row = words_per_row / 8u;
    float sum = 0.0f;

    for (uint input_base = 0u; input_base < input_width;
         input_base += QWEN38_DECODE_TILED_INPUTS) {
        const uint tile_elements = min(QWEN38_DECODE_TILED_INPUTS, input_width - input_base);
        for (uint index = thread_index; index < tile_elements;
             index += QWEN38_DECODE_SHARED_THREADS) {
            input_tile[index] = input[input_base + index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < output_rows) {
            const uint first_group = input_base / 64u;
            const uint groups_in_tile = tile_elements / 64u;
            for (uint group = first_group + lane;
                 group < first_group + groups_in_tile;
                 group += 32u) {
                const ulong parameter_offset = ulong(row * groups_per_row + group) * 2ul;
                const ulong scale_address = scale_byte_offset + parameter_offset;
                const ulong bias_address = bias_byte_offset + parameter_offset;
                const ushort scale_bits = ushort(scales[scale_address])
                    | (ushort(scales[scale_address + 1ul]) << 8u);
                const ushort bias_bits = ushort(biases[bias_address])
                    | (ushort(biases[bias_address + 1ul]) << 8u);
                const float scale = qwen38_bf16_to_float(scale_bits);
                const float bias = qwen38_bf16_to_float(bias_bits);
                const ulong word_base = weight_byte_offset
                    + ulong(row * words_per_row + group * 8u) * 4ul;
                const uint tile_offset = (group - first_group) * 64u;
                float dot_sum = 0.0f;
                float input_sum = 0.0f;
                for (uint word = 0; word < 8u; ++word) {
                    const ulong address = word_base + ulong(word) * 4ul;
                    const uint packed = uint(packed_weights[address])
                        | (uint(packed_weights[address + 1ul]) << 8u)
                        | (uint(packed_weights[address + 2ul]) << 16u)
                        | (uint(packed_weights[address + 3ul]) << 24u);
                    const uint offset = tile_offset + word * 8u;
                    dot_sum += qwen38_q4_word_dot_threadgroup(packed, input_tile, offset);
                    input_sum += input_tile[offset] + input_tile[offset + 1u]
                        + input_tile[offset + 2u] + input_tile[offset + 3u]
                        + input_tile[offset + 4u] + input_tile[offset + 5u]
                        + input_tile[offset + 6u] + input_tile[offset + 7u];
                }
                sum += dot_sum * scale + input_sum * bias;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum = simd_sum(sum);
    if (lane == 0u && row < output_rows) {
        output[row] = sum;
    }
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
#define QWEN38_PREFILL_THREADS 256u

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
             index += QWEN38_PREFILL_THREADS) {
            const uint staged_batch = tile.y * QWEN38_PREFILL_BATCH_TILE
                + index / QWEN38_PREFILL_AFFINE_TILE;
            const uint staged_element = affine_group * QWEN38_PREFILL_AFFINE_TILE
                + index % QWEN38_PREFILL_AFFINE_TILE;
            input_tile[index] = staged_batch < batch_size
                ? input[staged_batch * input_width + staged_element]
                : 0.0f;
        }
        if (thread_index < QWEN38_PREFILL_OUTPUT_TILE * QWEN38_PREFILL_WORDS_PER_AFFINE_TILE) {
            const uint packed_row = thread_index / QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
            const uint packed_word = thread_index % QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
            const uint staged_row = tile.x * QWEN38_PREFILL_OUTPUT_TILE + packed_row;
            packed_tile[thread_index] = staged_row < output_rows
                ? packed_weights[staged_row * words_per_row
                    + affine_group * QWEN38_PREFILL_WORDS_PER_AFFINE_TILE + packed_word]
                : 0u;
        }
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

        const uint input_base = local_batch * QWEN38_PREFILL_AFFINE_TILE;
        const uint lane = thread_index % QWEN38_PREFILL_OUTPUT_TILE;
        // Every lane must execute the SIMD reduction, including padded output
        // rows, otherwise edge tiles diverge and simd_sum is undefined.
        const float input_sum = simd_sum(
            input_tile[input_base + lane] + input_tile[input_base + lane + 32u]);
        if (batch < batch_size && row < output_rows) {
            const float scale = scale_tile[local_row];
            const float bias = bias_tile[local_row];
            const uint packed_base = local_row * QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
            float dot_sum = 0.0f;
            for (uint word = 0; word < QWEN38_PREFILL_WORDS_PER_AFFINE_TILE; ++word) {
                const uint packed = packed_tile[packed_base + word];
                dot_sum += qwen38_q4_word_dot_threadgroup(
                    packed, input_tile, input_base + word * 8u);
            }
            accumulator += dot_sum * scale + input_sum * bias;
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
             index += QWEN38_PREFILL_THREADS) {
            const uint staged_batch = tile.y * QWEN38_PREFILL_BATCH_TILE
                + index / QWEN38_PREFILL_AFFINE_TILE;
            const uint staged_element = affine_group * QWEN38_PREFILL_AFFINE_TILE
                + index % QWEN38_PREFILL_AFFINE_TILE;
            input_tile[index] = staged_batch < batch_size
                ? input[staged_batch * input_width + staged_element]
                : 0.0f;
        }
        if (thread_index < QWEN38_PREFILL_OUTPUT_TILE * QWEN38_PREFILL_WORDS_PER_AFFINE_TILE) {
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

        const uint input_base = local_batch * QWEN38_PREFILL_AFFINE_TILE;
        const uint lane = thread_index % QWEN38_PREFILL_OUTPUT_TILE;
        // Keep the reduction uniform for partial output tiles as above.
        const float input_sum = simd_sum(
            input_tile[input_base + lane] + input_tile[input_base + lane + 32u]);
        if (batch < batch_size && row < output_rows) {
            const float scale = scale_tile[local_row];
            const float bias = bias_tile[local_row];
            const uint packed_base = local_row * QWEN38_PREFILL_WORDS_PER_AFFINE_TILE;
            float dot_sum = 0.0f;
            for (uint word = 0; word < QWEN38_PREFILL_WORDS_PER_AFFINE_TILE; ++word) {
                const uint packed = packed_tile[packed_base + word];
                dot_sum += qwen38_q4_word_dot_threadgroup(
                    packed, input_tile, input_base + word * 8u);
            }
            accumulator += dot_sum * scale + input_sum * bias;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (batch < batch_size && row < output_rows) {
        output[batch * output_rows + row] = accumulator;
    }
}

// Long prompts need a different balance from decode: reuse every Q4 weight
// tile across 64 prompt rows and let each SIMD group execute an 8x8 matrix
// accumulation. Activations and the transient dequantized Q4 tile are half
// precision, while the accumulator and result remain F32. The generic tiled
// kernel remains the exact fallback for small batches and unaligned tensors.
#define QWEN38_SIMDGROUP_PREFILL_BATCH_TILE 64u
#define QWEN38_SIMDGROUP_PREFILL_OUTPUT_TILE 8u
#define QWEN38_SIMDGROUP_PREFILL_K_TILE 64u
#define QWEN38_SIMDGROUP_PREFILL_THREADS 256u

kernel void qwen38_q4_affine_matmul_simdgroup(
    device const float* input [[buffer(0)]],
    device const uint* packed_weights [[buffer(1)]],
    device const ushort* scales [[buffer(2)]],
    device const ushort* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& words_per_row [[buffer(5)]],
    constant uint& output_rows [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    threadgroup half* activation_tile [[threadgroup(0)]],
    threadgroup half* weight_tile [[threadgroup(1)]],
    threadgroup float* output_tile [[threadgroup(2)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint output_base = tile.x * QWEN38_SIMDGROUP_PREFILL_OUTPUT_TILE;
    const uint batch_base = tile.y * QWEN38_SIMDGROUP_PREFILL_BATCH_TILE;
    const uint input_width = words_per_row * 8u;
    const uint affine_groups_per_row = words_per_row / 8u;
    const uint simdgroup = thread_index / 32u;
    simdgroup_float8x8 accumulator = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint word_base = 0; word_base < words_per_row; word_base += 8u) {
        for (uint index = thread_index;
             index < QWEN38_SIMDGROUP_PREFILL_BATCH_TILE * QWEN38_SIMDGROUP_PREFILL_K_TILE;
             index += QWEN38_SIMDGROUP_PREFILL_THREADS) {
            const uint local_batch = index / QWEN38_SIMDGROUP_PREFILL_K_TILE;
            const uint local_element = index % QWEN38_SIMDGROUP_PREFILL_K_TILE;
            const uint batch = batch_base + local_batch;
            activation_tile[index] = batch < batch_size
                ? half(input[batch * input_width + word_base * 8u + local_element])
                : half(0.0h);
        }
        for (uint index = thread_index;
             index < QWEN38_SIMDGROUP_PREFILL_K_TILE * QWEN38_SIMDGROUP_PREFILL_OUTPUT_TILE;
             index += QWEN38_SIMDGROUP_PREFILL_THREADS) {
            const uint local_element = index / QWEN38_SIMDGROUP_PREFILL_OUTPUT_TILE;
            const uint local_output = index % QWEN38_SIMDGROUP_PREFILL_OUTPUT_TILE;
            const uint output_row = output_base + local_output;
            if (output_row < output_rows) {
                const uint packed_word = word_base + local_element / 8u;
                const uint packed = packed_weights[output_row * words_per_row + packed_word];
                const float quantized = float((packed >> ((local_element % 8u) * 4u)) & 0xFu);
                const uint affine_index = output_row * affine_groups_per_row + word_base / 8u;
                const float scale = qwen38_bf16_to_float(scales[affine_index]);
                const float bias = qwen38_bf16_to_float(biases[affine_index]);
                weight_tile[index] = half(quantized * scale + bias);
            } else {
                weight_tile[index] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_tile = 0; k_tile < 8u; ++k_tile) {
            simdgroup_half8x8 activation_matrix;
            simdgroup_half8x8 weight_matrix;
            simdgroup_load(
                activation_matrix,
                activation_tile,
                QWEN38_SIMDGROUP_PREFILL_K_TILE,
                ulong2(k_tile * 8u, simdgroup * 8u),
                false);
            simdgroup_load(
                weight_matrix,
                weight_tile,
                QWEN38_SIMDGROUP_PREFILL_OUTPUT_TILE,
                ulong2(0u, k_tile * 8u),
                false);
            simdgroup_multiply_accumulate(
                accumulator, activation_matrix, weight_matrix, accumulator);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(
        accumulator,
        output_tile,
        QWEN38_SIMDGROUP_PREFILL_OUTPUT_TILE,
        ulong2(0u, simdgroup * 8u),
        false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = thread_index;
         index < QWEN38_SIMDGROUP_PREFILL_BATCH_TILE * QWEN38_SIMDGROUP_PREFILL_OUTPUT_TILE;
         index += QWEN38_SIMDGROUP_PREFILL_THREADS) {
        const uint local_batch = index / QWEN38_SIMDGROUP_PREFILL_OUTPUT_TILE;
        const uint local_output = index % QWEN38_SIMDGROUP_PREFILL_OUTPUT_TILE;
        const uint batch = batch_base + local_batch;
        const uint output_row = output_base + local_output;
        if (batch < batch_size && output_row < output_rows) {
            output[batch * output_rows + output_row] = output_tile[index];
        }
    }
}

// Wider variant for long prompts. Thirty-two SIMD groups cover a 128x16
// output tile (each group owns an 8x8 submatrix), so the 64x64 activation
// staging is reused across twice as many output columns as the 64x8 kernel.
// The footprint is 16 KiB activations + 2 KiB weights + 8 KiB outputs.
#define QWEN38_SIMDGROUP_WIDE_BATCH_TILE 128u
#define QWEN38_SIMDGROUP_WIDE_OUTPUT_TILE 16u
#define QWEN38_SIMDGROUP_WIDE_K_TILE 64u
#define QWEN38_SIMDGROUP_WIDE_THREADS 1024u

kernel void qwen38_q4_affine_matmul_simdgroup_wide(
    device const float* input [[buffer(0)]],
    device const uint* packed_weights [[buffer(1)]],
    device const ushort* scales [[buffer(2)]],
    device const ushort* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& words_per_row [[buffer(5)]],
    constant uint& output_rows [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    threadgroup half* activation_tile [[threadgroup(0)]],
    threadgroup half* weight_tile [[threadgroup(1)]],
    threadgroup float* output_tile [[threadgroup(2)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint output_base = tile.x * QWEN38_SIMDGROUP_WIDE_OUTPUT_TILE;
    const uint batch_base = tile.y * QWEN38_SIMDGROUP_WIDE_BATCH_TILE;
    const uint input_width = words_per_row * 8u;
    const uint affine_groups_per_row = words_per_row / 8u;
    const uint simdgroup = thread_index / 32u;
    const uint output_block = simdgroup % 2u;
    const uint batch_block = simdgroup / 2u;
    simdgroup_float8x8 accumulator = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint word_base = 0; word_base < words_per_row; word_base += 8u) {
        for (uint index = thread_index;
             index < QWEN38_SIMDGROUP_WIDE_BATCH_TILE * QWEN38_SIMDGROUP_WIDE_K_TILE;
             index += QWEN38_SIMDGROUP_WIDE_THREADS) {
            const uint local_batch = index / QWEN38_SIMDGROUP_WIDE_K_TILE;
            const uint local_element = index % QWEN38_SIMDGROUP_WIDE_K_TILE;
            const uint batch = batch_base + local_batch;
            activation_tile[index] = batch < batch_size
                ? half(input[batch * input_width + word_base * 8u + local_element])
                : half(0.0h);
        }
        for (uint index = thread_index;
             index < QWEN38_SIMDGROUP_WIDE_K_TILE * QWEN38_SIMDGROUP_WIDE_OUTPUT_TILE;
             index += QWEN38_SIMDGROUP_WIDE_THREADS) {
            const uint local_element = index / QWEN38_SIMDGROUP_WIDE_OUTPUT_TILE;
            const uint local_output = index % QWEN38_SIMDGROUP_WIDE_OUTPUT_TILE;
            const uint output_row = output_base + local_output;
            if (output_row < output_rows) {
                const uint packed_word = word_base + local_element / 8u;
                const uint packed = packed_weights[output_row * words_per_row + packed_word];
                const float quantized = float((packed >> ((local_element % 8u) * 4u)) & 0xFu);
                const uint affine_index = output_row * affine_groups_per_row + word_base / 8u;
                const float scale = qwen38_bf16_to_float(scales[affine_index]);
                const float bias = qwen38_bf16_to_float(biases[affine_index]);
                weight_tile[index] = half(quantized * scale + bias);
            } else {
                weight_tile[index] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_tile = 0; k_tile < 8u; ++k_tile) {
            simdgroup_half8x8 activation_matrix;
            simdgroup_half8x8 weight_matrix;
            simdgroup_load(
                activation_matrix,
                activation_tile,
                QWEN38_SIMDGROUP_WIDE_K_TILE,
                ulong2(k_tile * 8u, batch_block * 8u),
                false);
            simdgroup_load(
                weight_matrix,
                weight_tile,
                QWEN38_SIMDGROUP_WIDE_OUTPUT_TILE,
                ulong2(output_block * 8u, k_tile * 8u),
                false);
            simdgroup_multiply_accumulate(
                accumulator, activation_matrix, weight_matrix, accumulator);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(
        accumulator,
        output_tile,
        QWEN38_SIMDGROUP_WIDE_OUTPUT_TILE,
        ulong2(output_block * 8u, batch_block * 8u),
        false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = thread_index;
         index < QWEN38_SIMDGROUP_WIDE_BATCH_TILE * QWEN38_SIMDGROUP_WIDE_OUTPUT_TILE;
         index += QWEN38_SIMDGROUP_WIDE_THREADS) {
        const uint local_batch = index / QWEN38_SIMDGROUP_WIDE_OUTPUT_TILE;
        const uint local_output = index % QWEN38_SIMDGROUP_WIDE_OUTPUT_TILE;
        const uint batch = batch_base + local_batch;
        const uint output_row = output_base + local_output;
        if (batch < batch_size && output_row < output_rows) {
            output[batch * output_rows + output_row] = output_tile[index];
        }
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

// Large prompt GEMMs use the platform's precompiled FP16 matrix kernel. Keep
// the Q4 model mapping compact and expand one projection into a reusable
// scratch buffer immediately before MPS consumes it. The logical matrix stays
// output-row major, allowing MPS to use it as a transposed right operand.
kernel void qwen38_q4_affine_dequantize_f16(
    device const uint* packed_weights [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& words_per_row [[buffer(4)]],
    constant uint& output_rows [[buffer(5)]],
    uint2 index [[thread_position_in_grid]]) {
    const uint word = index.x;
    const uint row = index.y;
    if (word >= words_per_row || row >= output_rows) {
        return;
    }
    const uint group = word / 8u;
    const uint groups_per_row = words_per_row / 8u;
    const float scale = qwen38_bf16_to_float(scales[row * groups_per_row + group]);
    const float bias = qwen38_bf16_to_float(biases[row * groups_per_row + group]);
    const uint packed = packed_weights[row * words_per_row + word];
    const uint output_base = row * words_per_row * 8u + word * 8u;
    for (uint nibble = 0; nibble < 8u; ++nibble) {
        const float quantized = float((packed >> (nibble * 4u)) & 0xFu);
        output[output_base + nibble] = half(quantized * scale + bias);
    }
}

kernel void qwen38_q4_affine_dequantize_f16_unaligned(
    device const uchar* packed_weights [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const uchar* biases [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& words_per_row [[buffer(4)]],
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant ulong& scale_byte_offset [[buffer(6)]],
    constant ulong& bias_byte_offset [[buffer(7)]],
    constant uint& output_rows [[buffer(8)]],
    uint2 index [[thread_position_in_grid]]) {
    const uint word = index.x;
    const uint row = index.y;
    if (word >= words_per_row || row >= output_rows) {
        return;
    }
    const uint group = word / 8u;
    const uint groups_per_row = words_per_row / 8u;
    const ulong parameter_offset = ulong(row * groups_per_row + group) * 2ul;
    const ulong scale_address = scale_byte_offset + parameter_offset;
    const ulong bias_address = bias_byte_offset + parameter_offset;
    const ushort scale_bits = ushort(scales[scale_address])
        | (ushort(scales[scale_address + 1ul]) << 8u);
    const ushort bias_bits = ushort(biases[bias_address])
        | (ushort(biases[bias_address + 1ul]) << 8u);
    const ulong weight_address = weight_byte_offset + ulong(row * words_per_row + word) * 4ul;
    const uint packed = uint(packed_weights[weight_address])
        | (uint(packed_weights[weight_address + 1ul]) << 8u)
        | (uint(packed_weights[weight_address + 2ul]) << 16u)
        | (uint(packed_weights[weight_address + 3ul]) << 24u);
    const float scale = qwen38_bf16_to_float(scale_bits);
    const float bias = qwen38_bf16_to_float(bias_bits);
    const uint output_base = row * words_per_row * 8u + word * 8u;
    for (uint nibble = 0; nibble < 8u; ++nibble) {
        const float quantized = float((packed >> (nibble * 4u)) & 0xFu);
        output[output_base + nibble] = half(quantized * scale + bias);
    }
}

kernel void qwen38_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& elements [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        output[index] = half(input[index]);
    }
}

kernel void qwen38_f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& elements [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        output[index] = float(input[index]);
    }
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

// Decode keeps the residual stream on the GPU across linear-attention
// layers. One threadgroup handles the small hidden vector, which avoids a
// host round-trip for the two RMSNorm operations in every layer.
kernel void qwen38_rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& elements [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    threadgroup float* partial [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint index = thread_index; index < elements; index += 256u) {
        const float value = input[index];
        sum += value * value;
    }
    partial[thread_index] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (thread_index < stride) {
            partial[thread_index] += partial[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(partial[0] / float(elements) + epsilon);
    for (uint index = thread_index; index < elements; index += 256u) {
        output[index] = input[index] * scale * weights[index];
    }
}

kernel void qwen38_add_in_place(
    device float* destination [[buffer(0)]],
    device const float* source [[buffer(1)]],
    constant uint& elements [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        destination[index] += source[index];
    }
}

kernel void qwen38_swiglu_half_rows(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        const float gate_value = float(gate[index]);
        output[index] = half(qwen38_silu(gate_value) * float(up[index]));
    }
}

// Gate and up can share one MPS result matrix. Each row stores gate followed
// by up, so this keeps the fused MLP chain on private GPU storage.
kernel void qwen38_swiglu_half_split_rows(
    device const half* gate_and_up [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& intermediate_width [[buffer(2)]],
    constant uint& elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        const uint row = index / intermediate_width;
        const uint column = index % intermediate_width;
        const uint base = row * intermediate_width * 2u + column;
        output[index] = half(qwen38_silu(float(gate_and_up[base]))
            * float(gate_and_up[base + intermediate_width]));
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

inline uint qwen38_mrope_axis(
    uint frequency,
    uint position0,
    uint position1,
    uint position2,
    uint section1,
    uint section2,
    uint has_sections) {
    if (has_sections == 0u) {
        return position0;
    }
    const uint axis = frequency % 3u;
    const uint axis_frequency = frequency / 3u;
    if (axis == 1u && axis_frequency < section1) {
        return position1;
    }
    if (axis == 2u && axis_frequency < section2) {
        return position2;
    }
    return position0;
}

// Qwen's full-attention projection packs query and output gate beside each
// other for every attention head. Normalize and rotate query in-place on the
// GPU so decode does not round-trip a 12,288-element projection through CPU.
kernel void qwen38_gqa_prepare_query(
    device const float* q_with_gate [[buffer(0)]],
    device const float* norm [[buffer(1)]],
    device float* query [[buffer(2)]],
    device float* gate [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& rotary_dim [[buffer(5)]],
    constant uint& position0 [[buffer(6)]],
    constant uint& position1 [[buffer(7)]],
    constant uint& position2 [[buffer(8)]],
    constant uint& section1 [[buffer(9)]],
    constant uint& section2 [[buffer(10)]],
    constant uint& has_sections [[buffer(11)]],
    constant float& rope_theta [[buffer(12)]],
    constant float& epsilon [[buffer(13)]],
    threadgroup float* partial [[threadgroup(0)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint input_base = head * head_dim * 2u;
    const uint output_base = head * head_dim;
    const float value = lane < head_dim ? q_with_gate[input_base + lane] : 0.0f;
    partial[lane] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < head_dim) {
        const float inverse = rsqrt(partial[0] / float(head_dim) + epsilon);
        const float normalized = value * inverse * norm[lane];
        float rotated = normalized;
        if (lane < rotary_dim) {
            const uint rotary_half = rotary_dim / 2u;
            const uint pair = lane < rotary_half ? lane : lane - rotary_half;
            const uint other_lane = lane < rotary_half ? lane + rotary_half : lane - rotary_half;
            const float other = q_with_gate[input_base + other_lane] * inverse * norm[other_lane];
            const uint axis = qwen38_mrope_axis(
                pair,
                position0,
                position1,
                position2,
                section1,
                section2,
                has_sections);
            const float exponent = float(pair * 2u) / float(rotary_dim);
            const float angle = float(axis) / pow(rope_theta, exponent);
            const float sine = sin(angle);
            const float cosine = cos(angle);
            rotated = lane < rotary_half ? normalized * cosine - other * sine
                : normalized * cosine + other * sine;
        }
        query[output_base + lane] = rotated;
        gate[output_base + lane] = q_with_gate[input_base + head_dim + lane];
    }
}

kernel void qwen38_gqa_prepare_key(
    device const float* key_input [[buffer(0)]],
    device const float* norm [[buffer(1)]],
    device float* key_output [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& rotary_dim [[buffer(4)]],
    constant uint& position0 [[buffer(5)]],
    constant uint& position1 [[buffer(6)]],
    constant uint& position2 [[buffer(7)]],
    constant uint& section1 [[buffer(8)]],
    constant uint& section2 [[buffer(9)]],
    constant uint& has_sections [[buffer(10)]],
    constant float& rope_theta [[buffer(11)]],
    constant float& epsilon [[buffer(12)]],
    threadgroup float* partial [[threadgroup(0)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint base = head * head_dim;
    const float value = lane < head_dim ? key_input[base + lane] : 0.0f;
    partial[lane] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < head_dim) {
        const float inverse = rsqrt(partial[0] / float(head_dim) + epsilon);
        const float normalized = value * inverse * norm[lane];
        float rotated = normalized;
        if (lane < rotary_dim) {
            const uint rotary_half = rotary_dim / 2u;
            const uint pair = lane < rotary_half ? lane : lane - rotary_half;
            const uint other_lane = lane < rotary_half ? lane + rotary_half : lane - rotary_half;
            const float other = key_input[base + other_lane] * inverse * norm[other_lane];
            const uint axis = qwen38_mrope_axis(
                pair,
                position0,
                position1,
                position2,
                section1,
                section2,
                has_sections);
            const float exponent = float(pair * 2u) / float(rotary_dim);
            const float angle = float(axis) / pow(rope_theta, exponent);
            const float sine = sin(angle);
            const float cosine = cos(angle);
            rotated = lane < rotary_half ? normalized * cosine - other * sine
                : normalized * cosine + other * sine;
        }
        key_output[base + lane] = rotated;
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
// Eight SIMD-groups score eight KV rows together, then one block update folds
// them into the running softmax. This avoids the per-token threadgroup
// barriers of the scalar online implementation while retaining exact causal
// visibility and Q8 KV storage.
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
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]) {
    constexpr uint key_block_tokens = 8u;
    const uint query_index = group / num_heads;
    const uint head = group % num_heads;
    const uint kv_head = head * kv_heads / num_heads;
    const uint query_offset = (query_index * num_heads + head) * head_dim;
    const uint visible_tokens = min(start_token + query_index + 1, total_length);
    threadgroup float* scores = scratch;
    threadgroup float* weights = scratch + key_block_tokens;
    threadgroup float* state = weights + key_block_tokens;
    float accumulator = 0.0f;
    if (thread_index == 0) {
        state[0] = -INFINITY;
        state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint token_base = 0; token_base < visible_tokens;
         token_base += key_block_tokens) {
        const uint token = token_base + simdgroup_index;
        float dot = 0.0f;
        if (token < visible_tokens) {
            const uint key_offset = (token * kv_heads + kv_head) * head_dim;
            const float key_scale = key_scales[token * kv_heads + kv_head];
            for (uint dimension = lane; dimension < head_dim; dimension += 32u) {
                dot += query[query_offset + dimension]
                    * float(keys[key_offset + dimension]) * key_scale;
            }
        }
        dot = simd_sum(dot);
        if (lane == 0u) {
            scores[simdgroup_index] = dot * rsqrt(float(head_dim));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index == 0) {
            float block_maximum = -INFINITY;
            for (uint index = 0; index < key_block_tokens; ++index) {
                if (token_base + index < visible_tokens) {
                    block_maximum = max(block_maximum, scores[index]);
                }
            }
            const float maximum = max(state[0], block_maximum);
            const float previous_scale = exp(state[0] - maximum);
            float block_sum = 0.0f;
            for (uint index = 0; index < key_block_tokens; ++index) {
                const float weight = token_base + index < visible_tokens
                    ? exp(scores[index] - maximum)
                    : 0.0f;
                weights[index] = weight;
                block_sum += weight;
            }
            state[0] = maximum;
            state[1] = state[1] * previous_scale + block_sum;
            state[2] = previous_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index < head_dim) {
            accumulator *= state[2];
            for (uint index = 0; index < key_block_tokens; ++index) {
                const uint value_token = token_base + index;
                if (value_token < visible_tokens) {
                    const uint value_offset = (value_token * kv_heads + kv_head) * head_dim
                        + thread_index;
                    accumulator += weights[index] * float(values[value_offset])
                        * value_scales[value_token * kv_heads + kv_head];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index < head_dim) {
        const uint output_offset = query_offset + thread_index;
        output[output_offset] = accumulator / max(state[1], FLT_MIN)
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
