# qwen38-metal

`qwen38-metal` is a native Apple Silicon runtime core for Qwen3.8 long-context work. It is written in Rust and embeds a build-time Metal library in the release binary. The distribution target is one native executable: no Python, MLX runtime, runtime shader compiler, or model-serving process.

This repository deliberately starts with the part that constrains the eventual engine: memory geometry, model compatibility checks, and native packaging. It does not load model weights or generate tokens yet, so it does not make an inference-speed claim.

## Runtime core

The current milestone provides an executable foundation for the eventual Qwen3.8-27B engine:

- A release binary with an embedded `MTLB` library compiled before Rust links the executable.
- A 48 GiB M4 Pro memory model for Qwen3.8-27B hybrid attention geometry.
- Paged KV cache planning for BF16, Q8, and Q4 precision.
- A logical page table that only allocates pages after tokens arrive.
- An adaptive one-to-three-token MTP depth controller.
- A model preflight that reads `config.json` and `model.safetensors.index.json`, then detects when a model declares MTP layers but the converted weights omit their tensors.

The next implementation milestone is a safetensors loader plus Metal execution for Q4e matmul, Gated DeltaNet, paged GQA attention, KV packing, and exact MTP verification. Those kernels are not claimed as implemented by this commit.

## Memory target

The default profile is one 262,144-token stream with Q8 paged KV. For the currently configured Qwen3.8-27B geometry, the KV data is 8 GiB plus about 1 MiB of per-page FP32 scales. The budgeting model reserves 17 GiB for mixed Q4 weights, 3 GiB for workspace, and 12 GiB for macOS and application headroom, leaving about 8 GiB under the 48 GiB unified-memory budget.

BF16 KV needs 16 GiB at the same context length and exceeds that planning budget. Q4 KV requires about 4 GiB before page scales, but is intended as an experimental capacity mode until its quality and kernel cost are measured.

## Commands

```text
qwen38-metal doctor
qwen38-metal plan --context 262144 --kv q8 --page-tokens 128
qwen38-metal preflight /path/to/qwen38-model
```

`doctor` checks the embedded Metal library and prints the default budget. `plan` accepts `bf16`, `q8`, or `q4`. `preflight` expects a Qwen3.5-style `config.json` and `model.safetensors.index.json`; it reports the detected attention geometry and whether native MTP tensors are present.

## CI contract

GitHub Actions is the validation authority for this early milestone. The `macos-14` ARM64 runner formats, lints, tests, compiles Metal with `xcrun`, builds the release binary, runs `doctor`, validates the native 262K/Q8 plan, and publishes the binary as an artifact.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
