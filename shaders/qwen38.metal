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
