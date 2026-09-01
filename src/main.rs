use qwen38_metal::api::{self, FixtureEngine, ServerConfig};
use qwen38_metal::geometry::{format_gib, CachePlan, KvPrecision, M4ProBudget, Qwen35Geometry};
use qwen38_metal::metal::embedded_library_info;
use qwen38_metal::metal_runtime::MetalRuntime;
use qwen38_metal::model::inspect_mlx_safetensors_dir;
use qwen38_metal::native::NativeEngine;
use qwen38_metal::native::NativeWeights;
use qwen38_metal::preflight::inspect_model_dir;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_CONTEXT_TOKENS: u32 = 262_144;
const DEFAULT_PAGE_TOKENS: u32 = 128;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = arguments.split_first() else {
        print_help();
        return Ok(());
    };

    match command.as_str() {
        "help" | "--help" | "-h" => {
            require_no_arguments(command, rest)?;
            print_help();
            Ok(())
        }
        "version" | "--version" | "-V" => {
            require_no_arguments(command, rest)?;
            println!("qwen38-metal {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "doctor" => {
            require_no_arguments(command, rest)?;
            print_doctor()
        }
        "plan" => print_plan(PlanOptions::parse(rest)?),
        "serve" => serve(ServeOptions::parse(rest)?),
        "q4-probe" => run_q4_probe(Q4ProbeOptions::parse(rest)?),
        "inspect-model" => {
            let model_dir = match rest {
                [model_dir] => model_dir,
                _ => return Err("usage: qwen38-metal inspect-model <MODEL_DIR>".to_owned()),
            };
            print_model_manifest(model_dir)
        }
        "preflight" => {
            let model_dir = match rest {
                [model_dir] => model_dir,
                _ => return Err("usage: qwen38-metal preflight <MODEL_DIR>".to_owned()),
            };
            print_preflight(model_dir)
        }
        _ => Err(format!(
            "unknown command {command:?}; run `qwen38-metal help` for usage"
        )),
    }
}

fn require_no_arguments(command: &str, arguments: &[String]) -> Result<(), String> {
    if arguments.is_empty() {
        Ok(())
    } else {
        Err(format!("{command} does not accept arguments"))
    }
}

fn print_help() {
    println!(
        "qwen38-metal\n\n\
         Native Rust and precompiled Metal runtime core for Qwen3.8 long-context inference.\n\n\
         Usage:\n\
           qwen38-metal doctor\n\
           qwen38-metal plan [--context TOKENS] [--kv bf16|q8|q4] [--page-tokens TOKENS]\n\
           qwen38-metal serve --model MODEL_DIR [--mtp-adapter ADAPTER_DIR] [--model-id ID] [--host IP] [--port PORT] [--generation-concurrency COUNT] [--max-queued-requests COUNT]\n\
           qwen38-metal serve --fixture-response TEXT [--model-id ID] [--host IP] [--port PORT] [--generation-concurrency COUNT] [--max-queued-requests COUNT]\n\
           qwen38-metal q4-probe MODEL_DIR [--tensor NAME] [--iterations COUNT] [--batch TOKENS]\n\
           qwen38-metal inspect-model MODEL_DIR\n\
           qwen38-metal preflight MODEL_DIR\n\
           qwen38-metal version\n\n\
         `inspect-model` validates MLX safetensors headers and affine quantization groups.\n\
         `serve --model` runs native Qwen inference and exposes OpenAI and Anthropic APIs.\n\
         `--fixture-response` is only for protocol validation. `plan` reports a 48 GiB M4 Pro\n\
         memory budget. `q4-probe` validates MLX Q4 Metal execution for one projection.\n\
         `preflight` detects native MTP weights. Set `--mtp-adapter` (or `QWEN38_MTP_ADAPTER`)\n\
         to enable the matching standalone Qwen MTP drafter for deterministic decoding. Native inference defaults to one generation\n\
         lane and a 64-request bounded queue; increase lanes only after measuring the target\n\
         workload because this runtime uses latency-oriented per-request state instead of continuous batching."
    );
}

fn run_q4_probe(options: Q4ProbeOptions) -> Result<(), String> {
    let runtime = MetalRuntime::new().map_err(|error| error.to_string())?;
    let weights =
        NativeWeights::open(&options.model_dir, &runtime).map_err(|error| error.to_string())?;
    let matrix = weights
        .q4_matrix(&options.tensor)
        .map_err(|error| error.to_string())?;
    let input_elements = usize::try_from(matrix.input_elements)
        .map_err(|_| "matrix input dimension exceeds host limits".to_owned())?;
    let input_len = input_elements
        .checked_mul(options.batch)
        .ok_or_else(|| "Q4 probe input size overflows host limits".to_owned())?;
    let input: Vec<f32> = (0..input_len)
        .map(|index| ((index % 29) as f32 - 14.0) / 29.0)
        .collect();
    let run_projection = || -> Result<Vec<f32>, String> {
        let output = if options.batch == 1 {
            weights
                .q4_affine_matvec(&runtime, &matrix, &input)
                .map_err(|error| error.to_string())?
        } else {
            weights
                .q4_affine_matmul_batch(&runtime, &[&matrix], &input, options.batch)
                .map_err(|error| error.to_string())?
                .remove(0)
        };
        Ok(output)
    };
    let started = Instant::now();
    let mut output = Vec::new();
    for _ in 0..options.iterations {
        output = run_projection()?;
    }
    let elapsed = started.elapsed();
    let checksum = output.iter().map(|value| f64::from(*value)).sum::<f64>();
    println!("execution: native Metal Q4 affine matvec");
    println!(
        "mapped safetensors shards: {}",
        weights.mapped_shard_count()
    );
    println!("projection: {}", options.tensor);
    println!(
        "matrix: {} output rows x {} input elements",
        matrix.output_rows, matrix.input_elements
    );
    println!("iterations: {}", options.iterations);
    println!("batch tokens: {}", options.batch);
    println!("elapsed: {:.3} ms", elapsed.as_secs_f64() * 1_000.0);
    println!("output checksum: {checksum:.6}");
    Ok(())
}

fn serve(options: ServeOptions) -> Result<(), String> {
    if !options.host.is_loopback() && !options.allow_remote {
        return Err(
            "refusing a non-loopback bind without --allow-remote; the default is 127.0.0.1"
                .to_owned(),
        );
    }
    if !options.host.is_loopback() && options.api_key.is_none() {
        return Err(
            "a non-loopback bind requires --api-key or --api-key-env to protect the local model"
                .to_owned(),
        );
    }
    let (engine, execution_label): (Arc<dyn qwen38_metal::api::InferenceEngine>, &str) =
        if let Some(model_dir) = &options.model_dir {
            let engine = match options.mtp_adapter.as_deref() {
                Some(adapter) => NativeEngine::open_with_mtp(
                    model_dir,
                    options.model_id,
                    Some(std::path::Path::new(adapter)),
                ),
                None => NativeEngine::open(model_dir, options.model_id),
            }
            .map_err(|error| error.to_string())?;
            (Arc::new(engine), "native")
        } else {
            let response = options.fixture_response.ok_or_else(|| {
                "a model directory is required for native execution; --fixture-response is available only for protocol validation"
                    .to_owned()
            })?;
            (
                Arc::new(FixtureEngine::new(
                    options.model_id,
                    options.context_tokens,
                    response,
                )),
                "fixture",
            )
        };
    let config = ServerConfig {
        max_output_tokens: options.max_output_tokens,
        api_key: options.api_key,
        max_request_bytes: 8 * 1024 * 1024,
        generation_concurrency: options.generation_concurrency,
        max_queued_requests: options.max_queued_requests,
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("cannot create HTTP runtime: {error}"))?;

    if execution_label == "fixture" {
        eprintln!(
            "warning: fixture execution is active; this validates client compatibility but does not run Qwen weights"
        );
    }
    runtime
        .block_on(api::serve(
            SocketAddr::new(options.host, options.port),
            engine,
            config,
        ))
        .map_err(|error| error.to_string())
}

fn print_doctor() -> Result<(), String> {
    let metal = embedded_library_info().map_err(|error| error.to_string())?;

    println!("binary: qwen38-metal {}", env!("CARGO_PKG_VERSION"));
    println!("runtime: native Rust");
    println!("Metal source: compiled before Rust links the release binary");
    println!("embedded metallib: {} bytes", metal.byte_len);
    println!("default profile: 262144-token context, Q8 paged KV");
    print_memory_plan(
        &Qwen35Geometry::qwen38_27b(),
        DEFAULT_CONTEXT_TOKENS,
        KvPrecision::Q8,
        DEFAULT_PAGE_TOKENS,
    )
}

fn print_plan(options: PlanOptions) -> Result<(), String> {
    print_memory_plan(
        &Qwen35Geometry::qwen38_27b(),
        options.context_tokens,
        options.precision,
        options.page_tokens,
    )
}

fn print_preflight(model_dir: &str) -> Result<(), String> {
    let inspection = inspect_model_dir(model_dir).map_err(|error| error.to_string())?;
    let architectures = if inspection.architectures.is_empty() {
        "(not declared)".to_owned()
    } else {
        inspection.architectures.join(", ")
    };
    let context_tokens = inspection
        .max_context_tokens
        .unwrap_or(DEFAULT_CONTEXT_TOKENS);

    println!("architectures: {architectures}");
    println!("declared max context: {context_tokens} tokens");
    println!(
        "attention geometry: {} full-attention layers, {} KV heads, head dim {}",
        inspection.geometry.full_attention_layers,
        inspection.geometry.num_key_value_heads,
        inspection.geometry.head_dim
    );
    println!("native MTP: {}", inspection.mtp_support);
    print_memory_plan(
        &inspection.geometry,
        context_tokens,
        KvPrecision::Q8,
        DEFAULT_PAGE_TOKENS,
    )
}

fn print_model_manifest(model_dir: &str) -> Result<(), String> {
    let manifest = inspect_mlx_safetensors_dir(model_dir).map_err(|error| error.to_string())?;
    let architectures = if manifest.architectures.is_empty() {
        "(not declared)".to_owned()
    } else {
        manifest.architectures.join(", ")
    };

    println!("format: {}", manifest.format);
    println!("architectures: {architectures}");
    println!("model type: {}", manifest.model_type);
    println!(
        "quantization: {}-bit {}, group size {}",
        manifest.quantization.bits, manifest.quantization.mode, manifest.quantization.group_size
    );
    println!("safetensors shards: {}", manifest.shard_count);
    println!("indexed tensors: {}", manifest.indexed_tensor_count);
    println!(
        "indexed tensor data: {}",
        format_gib(manifest.indexed_tensor_bytes)
    );
    println!(
        "affine quantized tensor groups: {}",
        manifest.quantized_tensor_groups
    );
    println!(
        "attention geometry: {} full-attention layers, {} KV heads, head dim {}",
        manifest.geometry.full_attention_layers,
        manifest.geometry.num_key_value_heads,
        manifest.geometry.head_dim
    );
    println!("native MTP: {}", manifest.mtp_support);
    Ok(())
}

fn print_memory_plan(
    geometry: &Qwen35Geometry,
    context_tokens: u32,
    precision: KvPrecision,
    page_tokens: u32,
) -> Result<(), String> {
    let cache = CachePlan::new(geometry, precision, context_tokens, page_tokens)
        .map_err(|error| error.to_string())?;
    let budget = M4ProBudget::default();
    let report = budget.report(&cache).map_err(|error| error.to_string())?;

    println!("KV precision: {}", cache.precision.label());
    println!("context: {} tokens", cache.context_tokens);
    println!(
        "paged KV: {} pages x {} tokens",
        cache.page_count, cache.page_tokens
    );
    println!("KV data: {}", format_gib(cache.data_bytes));
    println!("KV page scales: {}", format_gib(cache.page_scale_bytes));
    println!("KV total: {}", format_gib(cache.total_bytes));
    println!(
        "M4 Pro 48 GiB budget: {}",
        format_gib(budget.unified_memory_bytes)
    );
    println!("estimated total: {}", format_gib(report.required_bytes));

    match report.headroom_bytes {
        Some(headroom) => println!("estimated headroom: {}", format_gib(headroom)),
        None => println!(
            "estimated overage: {}",
            format_gib(report.required_bytes - report.available_bytes)
        ),
    }

    println!(
        "fits M4 Pro 48 GiB budget: {}",
        if report.fits() { "yes" } else { "no" }
    );
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct PlanOptions {
    context_tokens: u32,
    precision: KvPrecision,
    page_tokens: u32,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            context_tokens: DEFAULT_CONTEXT_TOKENS,
            precision: KvPrecision::Q8,
            page_tokens: DEFAULT_PAGE_TOKENS,
        }
    }
}

impl PlanOptions {
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut options = Self::default();
        let mut index = 0;

        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("{flag} requires a value"))?;
            index += 2;

            match flag.as_str() {
                "--context" => {
                    options.context_tokens = parse_positive_u32("--context", value)?;
                }
                "--kv" => {
                    options.precision =
                        KvPrecision::parse(value).map_err(|error| error.to_string())?;
                }
                "--page-tokens" => {
                    options.page_tokens = parse_positive_u32("--page-tokens", value)?;
                }
                _ => return Err(format!("unknown plan option {flag:?}")),
            }
        }

        Ok(options)
    }
}

fn parse_positive_u32(flag: &str, value: &str) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("{flag} requires a positive unsigned integer, got {value:?}"))?;
    if parsed == 0 {
        return Err(format!(
            "{flag} requires a positive unsigned integer, got {value:?}"
        ));
    }

    Ok(parsed)
}

#[derive(Debug)]
struct ServeOptions {
    host: IpAddr,
    port: u16,
    model_id: String,
    model_dir: Option<String>,
    mtp_adapter: Option<String>,
    context_tokens: u32,
    max_output_tokens: u32,
    api_key: Option<String>,
    fixture_response: Option<String>,
    allow_remote: bool,
    generation_concurrency: usize,
    max_queued_requests: usize,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".parse().expect("loopback address is valid"),
            port: 8_000,
            model_id: "qwen3.8-27b".to_owned(),
            model_dir: None,
            mtp_adapter: None,
            context_tokens: DEFAULT_CONTEXT_TOKENS,
            max_output_tokens: 4_096,
            api_key: None,
            fixture_response: None,
            allow_remote: false,
            generation_concurrency: 1,
            max_queued_requests: 64,
        }
    }
}

impl ServeOptions {
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut options = Self::default();
        let mut api_key_env = None;
        let mut index = 0;

        while index < arguments.len() {
            let flag = &arguments[index];
            if flag == "--allow-remote" {
                options.allow_remote = true;
                index += 1;
                continue;
            }

            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("{flag} requires a value"))?;
            index += 2;
            match flag.as_str() {
                "--host" => {
                    options.host = value
                        .parse::<IpAddr>()
                        .map_err(|_| format!("--host requires an IP address, got {value:?}"))?;
                }
                "--port" => {
                    options.port = value.parse::<u16>().map_err(|_| {
                        format!("--port requires an integer between 1 and 65535, got {value:?}")
                    })?;
                    if options.port == 0 {
                        return Err(
                            "--port requires an integer between 1 and 65535, got 0".to_owned()
                        );
                    }
                }
                "--model" => options.model_dir = Some(value.clone()),
                "--mtp-adapter" => {
                    if value.is_empty() {
                        return Err("--mtp-adapter cannot be empty".to_owned());
                    }
                    options.mtp_adapter = Some(value.clone());
                }
                "--model-id" => {
                    if value.is_empty() {
                        return Err("--model-id cannot be empty".to_owned());
                    }
                    options.model_id = value.clone();
                }
                "--context" => options.context_tokens = parse_positive_u32("--context", value)?,
                "--max-output-tokens" => {
                    options.max_output_tokens = parse_positive_u32("--max-output-tokens", value)?
                }
                "--api-key" => {
                    if value.is_empty() {
                        return Err("--api-key cannot be empty".to_owned());
                    }
                    options.api_key = Some(value.clone());
                }
                "--api-key-env" => {
                    if value.is_empty() {
                        return Err("--api-key-env cannot be empty".to_owned());
                    }
                    api_key_env = Some(value.clone());
                }
                "--fixture-response" => options.fixture_response = Some(value.clone()),
                "--generation-concurrency" => {
                    options.generation_concurrency =
                        parse_positive_usize("--generation-concurrency", value)?
                }
                "--max-queued-requests" => {
                    options.max_queued_requests =
                        parse_positive_usize("--max-queued-requests", value)?
                }
                _ => return Err(format!("unknown serve option {flag:?}")),
            }
        }

        if options.api_key.is_some() && api_key_env.is_some() {
            return Err("use either --api-key or --api-key-env, not both".to_owned());
        }
        if let Some(name) = api_key_env {
            let value = std::env::var(&name)
                .map_err(|_| format!("environment variable {name:?} is not set"))?;
            if value.is_empty() {
                return Err(format!("environment variable {name:?} is empty"));
            }
            options.api_key = Some(value);
        }
        if options.max_output_tokens > options.context_tokens {
            return Err(
                "--max-output-tokens cannot exceed --context because every request reserves its requested output"
                    .to_owned(),
            );
        }
        if options.max_queued_requests < options.generation_concurrency {
            return Err(
                "--max-queued-requests must be at least --generation-concurrency because active requests count toward the queue limit"
                    .to_owned(),
            );
        }
        Ok(options)
    }
}

fn parse_positive_usize(flag: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{flag} requires a positive unsigned integer, got {value:?}"))?;
    if parsed == 0 {
        return Err(format!(
            "{flag} requires a positive unsigned integer, got {value:?}"
        ));
    }
    Ok(parsed)
}

#[derive(Debug)]
struct Q4ProbeOptions {
    model_dir: String,
    tensor: String,
    iterations: u32,
    batch: usize,
}

impl Q4ProbeOptions {
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let Some((model_dir, rest)) = arguments.split_first() else {
            return Err(
                "usage: qwen38-metal q4-probe MODEL_DIR [--tensor NAME] [--iterations COUNT] [--batch TOKENS]"
                    .to_owned(),
            );
        };
        let mut options = Self {
            model_dir: model_dir.clone(),
            tensor: "language_model.model.layers.0.linear_attn.in_proj_b.weight".to_owned(),
            iterations: 1,
            batch: 1,
        };
        let mut index = 0;
        while index < rest.len() {
            let flag = &rest[index];
            let value = rest
                .get(index + 1)
                .ok_or_else(|| format!("{flag} requires a value"))?;
            index += 2;
            match flag.as_str() {
                "--tensor" => options.tensor = value.clone(),
                "--iterations" => options.iterations = parse_positive_u32("--iterations", value)?,
                "--batch" => options.batch = parse_positive_usize("--batch", value)?,
                _ => return Err(format!("unknown q4-probe option {flag:?}")),
            }
        }
        Ok(options)
    }
}
