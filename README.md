# qwen38-metal

`qwen38-metal` is a native Apple Silicon inference runtime for Qwen3.8 long-context work. The engine is written in Rust and ships precompiled Metal libraries inside its binary. It does not require Python, MLX, a runtime shader compiler, or a model-serving process.

The project is at the runtime-substrate milestone. It proves the distribution shape first: a single native binary embeds a build-time `metallib`, and GitHub Actions builds and runs it on macOS ARM64. It does not claim to run Qwen3.8 yet.

## Design target

- Qwen3.8-27B mixed 4-bit weights with matching native MTP heads.
- Native 262,144-token context, one active long-context stream.
- Paged INT8 KV cache by default. Q4 KV is an experimental memory-saving mode.
- Precompiled Metal kernels for Q4e matmul, Gated DeltaNet, paged GQA attention, KV packing, and exact MTP verification.
- Immutable prefix caching for large documents; SSD is a cold-cache tier only, never the active decode cache.

## Performance hypotheses

The figures below are engineering targets for an M4 Pro with 20 GPU cores and 48 GB unified memory. They are not measured results from this repository.

| Context | Target generation throughput |
| --- | ---: |
| 1K | 32-36 tok/s |
| 4K | 26-30 tok/s |
| 64K | 17-20 tok/s |
| 128K | 14-17 tok/s |
| 262K | 10-13 tok/s |

The 262K profile requires an INT8 paged KV cache. BF16 KV alone is about 16 GiB at this context length and leaves insufficient headroom on a 48 GB Mac.

## Build contract

GitHub Actions is the source of validation for this milestone. The workflow runs on `macos-14`, compiles Metal source with `xcrun`, links the Metal library before Rust links the executable, and runs `qwen38-metal doctor` from the produced ARM64 artifact.

## License

GPL-3.0-only. The repository is created with GitHub's GPL-3.0 license template.
