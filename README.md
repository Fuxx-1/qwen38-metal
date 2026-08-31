# qwen38-metal

`qwen38-metal` is a native Apple Silicon runtime for Qwen3.8 MLX 4-bit models. It is written in Rust and embeds a build-time Metal library in one executable: no Python, MLX runtime, runtime shader compiler, or model-serving process is needed.

The runtime loads Qwen3.8 MLX affine-Q4 safetensors, executes the hybrid DeltaNet/GQA model path on Metal, and serves native OpenAI Chat Completions and Anthropic Messages APIs. It accepts image inputs, client-managed function/tool calls, bounded concurrent requests, and Anthropic extended thinking. Audio input is not implemented.

## Run a native model

Build an ARM64 release binary, then point it at an MLX 4-bit Qwen3.8 model directory:

```text
cargo build --release
target/release/qwen38-metal serve \
  --model /path/to/Qwen3.8-27B-MLX-4bit \
  --model-id qwen3.8-27b-native \
  --generation-concurrency 1 \
  --max-queued-requests 64
```

The server binds to `127.0.0.1:8000` by default. A non-loopback bind requires both `--allow-remote` and `--api-key` or `--api-key-env`, preventing an unprotected local model from being exposed accidentally.

`--fixture-response` starts a protocol-only fixture server and never loads model weights. It is intended for CI and client integration tests, not production inference. `--generation-concurrency` controls how many requests may execute at once; `--max-queued-requests` includes active requests and bounds the waiting queue. Native Qwen defaults to one lane because recurrent state is latency-oriented; it already batches same-input projections and keeps per-request state isolated. Increasing lanes is supported but must be measured for the target workload because it can reduce per-request throughput.

## Compatible endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`, including Server-Sent Events, `stream_options.include_usage`, function tools, tool-result history, and user `image_url` blocks
- `POST /v1/messages`, including Anthropic named Server-Sent Events, `text`, `image`, `tool_use`, `tool_result`, and extended `thinking` blocks

OpenAI accepts `system`, `developer`, `user`, `assistant`, and `tool` messages. Anthropic requires the `anthropic-version` header and accepts `system`, `user`, and `assistant` messages. Image inputs may be base64 data, or a public `http(s)` URL; downloads reject local and private network targets, cap compressed bytes at 16 MiB, and validate decoded dimensions before inference. Tool execution remains with the API client: the server emits a requested call, then accepts the client-provided result in a later request.

Both APIs return standard SSE framing when `stream` is enabled. Native decoding forwards each decoded token immediately: OpenAI receives content/reasoning deltas, and Anthropic receives incremental `thinking_delta` or `text_delta` blocks. Tool-call markup is withheld until it has been parsed into a structured call, so clients never receive the model's internal XML protocol as visible text.

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

Example OpenAI tool request:

```text
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b-native","messages":[{"role":"user","content":"Weather in Shanghai?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"required","max_tokens":128}'
```

Example Anthropic extended-thinking request:

```text
curl http://127.0.0.1:8000/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"qwen3.8-27b-native","max_tokens":256,"thinking":{"type":"enabled","budget_tokens":128},"messages":[{"role":"user","content":"Explain 2 plus 2 briefly."}]}'
```

## Runtime design

Weights remain file-mapped rather than being expanded to floating point. The fast Q4 matvec kernel reads aligned `U32` and `BF16` safetensor regions directly. Q/K/V, DeltaNet's four input projections, and MLP gate/up projections share one activation upload and command buffer; reusable shared buffers avoid steady-state allocation churn. Safetensors headers are not required to leave their data region aligned, so the runtime automatically switches only affected projections to a byte-addressed Metal kernel. That path keeps the same mapped bytes and avoids a second copy of a multi-gigabyte shard.

DeltaNet convolution, q/k preparation, recurrent state update, and output gating run in the precompiled Metal library. Full GQA attention uses dynamically growing Q8 KV buffers with per-token/head scales and keeps attention score/value work on Metal. Prompt prefill is layer-major: tiled Q4 kernels process prompt-row and output-row blocks together, while the MLP's gate/up, SwiGLU, and down projection share one command buffer. DeltaNet and GQA advance the complete causal span in GPU kernels instead of submitting a command buffer for every position, and active KV capacity is reserved without allocating 262K pages at request start.

The published Qwen3.8 MLX 4-bit export declares one MTP layer but omits its tensors. The runtime therefore exposes speculative decoding as unavailable and executes verified standard decode. It will not claim MTP speedup until a matching verifier and proposer are loaded.

## Memory target

The default profile is one 262,144-token stream with Q8 paged KV. For the currently configured Qwen3.8-27B geometry, the KV data is 8 GiB plus about 1 MiB of per-page FP32 scales. The budgeting model reserves 17 GiB for mixed Q4 weights, 3 GiB for workspace, and 12 GiB for macOS and application headroom, leaving about 8 GiB under the 48 GiB unified-memory budget.

BF16 KV needs 16 GiB at the same context length and exceeds that planning budget. Q4 KV requires about 4 GiB before page scales, but is intended as an experimental capacity mode until its quality and kernel cost are measured. The plan is a capacity model, not a substitute for measuring the exact model, context length, and request shape before setting a production limit. The native runtime begins with small Q8 allocations and doubles active KV capacity only when a sequence reaches the current bound.

Text-only prompts also use an in-process longest-prefix cache. It stores up to two prefixes and 65,536 tokens in total by default, and only caches prefixes of at least 64 tokens. Cached entries own copies of the mutable DeltaNet and full-attention Q8 KV state, so a 65K-token entry costs roughly the Q8 KV footprint for that prefix (about 2 GiB for the published geometry). Set `QWEN38_PREFIX_CACHE_MAX_ENTRIES=0` to disable it, or tune `QWEN38_PREFIX_CACHE_MAX_ENTRIES`, `QWEN38_PREFIX_CACHE_MAX_TOKENS`, and `QWEN38_PREFIX_CACHE_MIN_TOKENS` for a tighter memory budget. Image requests bypass this token-only cache.

## Commands

```text
qwen38-metal doctor
qwen38-metal plan --context 262144 --kv q8 --page-tokens 128
qwen38-metal inspect-model /path/to/qwen38-mlx-model
qwen38-metal preflight /path/to/qwen38-model
qwen38-metal q4-probe /path/to/qwen38-mlx-model
qwen38-metal serve --model /path/to/qwen38-mlx-model --generation-concurrency 2 --max-queued-requests 8
```

`doctor` checks the embedded Metal library and prints the default budget. `plan` accepts `bf16`, `q8`, or `q4`. `inspect-model` validates MLX 4-bit affine safetensors metadata and reports its shard/tensor inventory. `preflight` expects a Qwen3.5-style `config.json` and `model.safetensors.index.json`; it reports the detected attention geometry and whether native MTP tensors are present. `q4-probe` runs one actual projection through the mapped Metal path.

## CI contract

GitHub Actions runs on an ARM64 macOS runner. It formats, lints, tests both Metal Q4 paths, compiles Metal with `xcrun`, builds the release binary, checks the 262K/Q8 plan, and starts fixture servers to smoke-test OpenAI and Anthropic text, SSE, images, tools, and thinking contracts. Native model execution is additionally validated locally because model files are not part of the repository or CI artifact.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
