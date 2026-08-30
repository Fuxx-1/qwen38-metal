# qwen38-metal

`qwen38-metal` is a native Apple Silicon runtime for Qwen3.8 MLX 4-bit models. It is written in Rust and embeds a build-time Metal library in one executable: no Python, MLX runtime, runtime shader compiler, or model-serving process is needed.

The runtime loads Qwen3.8 MLX affine-Q4 safetensors, executes the hybrid DeltaNet/GQA model path on Metal, and serves native OpenAI Chat Completions and Anthropic Messages APIs. It is intentionally text-only: image, audio, tool/function calling, and Anthropic extended-thinking requests return explicit unsupported-request errors.

## Run a native model

Build an ARM64 release binary, then point it at an MLX 4-bit Qwen3.8 model directory:

```text
cargo build --release
target/release/qwen38-metal serve \
  --model /path/to/Qwen3.8-27B-MLX-4bit \
  --model-id qwen3.8-27b-native
```

The server binds to `127.0.0.1:8000` by default. A non-loopback bind requires both `--allow-remote` and `--api-key` or `--api-key-env`, preventing an unprotected local model from being exposed accidentally.

`--fixture-response` starts a protocol-only fixture server and never loads model weights. It is intended for CI and client integration tests, not production inference.

## Compatible endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`, including Server-Sent Events and `stream_options.include_usage`
- `POST /v1/messages`, including Anthropic named Server-Sent Events

OpenAI accepts `system`, `developer`, `user`, and `assistant` messages. Anthropic requires the `anthropic-version` header and accepts `system`, `user`, and `assistant` text content. Both interfaces accept string text and standard text-content arrays. Each server instance uses one generation lane, so a concurrent request receives a retryable `429` rather than competing for the model state.

Example OpenAI request:

```text
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b-native","messages":[{"role":"user","content":"What is 2 plus 2?"}],"max_tokens":32}'
```

Example Anthropic request:

```text
curl http://127.0.0.1:8000/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"qwen3.8-27b-native","max_tokens":32,"messages":[{"role":"user","content":"What is 2 plus 2?"}]}'
```

## Runtime design

Weights remain file-mapped rather than being expanded to floating point. The fast Q4 matvec kernel reads aligned `U32` and `BF16` safetensor regions directly. Safetensors headers are not required to leave their data region aligned, so the runtime automatically switches only affected projections to a byte-addressed Metal kernel. That path keeps the same mapped bytes and avoids a second copy of a multi-gigabyte shard.

## Memory target

The default profile is one 262,144-token stream with Q8 paged KV. For the currently configured Qwen3.8-27B geometry, the KV data is 8 GiB plus about 1 MiB of per-page FP32 scales. The budgeting model reserves 17 GiB for mixed Q4 weights, 3 GiB for workspace, and 12 GiB for macOS and application headroom, leaving about 8 GiB under the 48 GiB unified-memory budget.

BF16 KV needs 16 GiB at the same context length and exceeds that planning budget. Q4 KV requires about 4 GiB before page scales, but is intended as an experimental capacity mode until its quality and kernel cost are measured. The plan is a capacity model, not a substitute for measuring the exact model, context length, and request shape before setting a production limit.

## Commands

```text
qwen38-metal doctor
qwen38-metal plan --context 262144 --kv q8 --page-tokens 128
qwen38-metal inspect-model /path/to/qwen38-mlx-model
qwen38-metal preflight /path/to/qwen38-model
qwen38-metal q4-probe /path/to/qwen38-mlx-model
```

`doctor` checks the embedded Metal library and prints the default budget. `plan` accepts `bf16`, `q8`, or `q4`. `inspect-model` validates MLX 4-bit affine safetensors metadata and reports its shard/tensor inventory. `preflight` expects a Qwen3.5-style `config.json` and `model.safetensors.index.json`; it reports the detected attention geometry and whether native MTP tensors are present. `q4-probe` runs one actual projection through the mapped Metal path.

## CI contract

GitHub Actions runs on an ARM64 macOS runner. It formats, lints, tests both Metal Q4 paths, compiles Metal with `xcrun`, builds the release binary, checks the 262K/Q8 plan, and starts a fixture server to smoke-test the OpenAI and Anthropic HTTP/SSE contracts. Native model execution is additionally validated locally because model files are not part of the repository or CI artifact.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
