use qwen38_metal::geometry::{
    format_gib, CachePlan, KvPrecision, M4ProBudget, Qwen35Geometry,
};
use qwen38_metal::metal::embedded_library_info;
use qwen38_metal::preflight::inspect_model_dir;

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
           qwen38-metal preflight MODEL_DIR\n\
           qwen38-metal version\n\n\
         `plan` reports a 48 GiB M4 Pro memory budget. `preflight` reads a Qwen3.5-style\n\
         config.json and model.safetensors.index.json to detect native MTP weights."
    );
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
    println!("M4 Pro 48 GiB budget: {}", format_gib(budget.unified_memory_bytes));
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
