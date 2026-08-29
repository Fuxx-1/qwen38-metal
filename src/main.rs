const METALLIB: &[u8] = include_bytes!(env!("QWEN38_METALLIB"));

fn main() {
    match std::env::args().nth(1).as_deref() {
        None | Some("help") | Some("--help") | Some("-h") => print_help(),
        Some("doctor") => print_doctor(),
        Some("version") | Some("--version") | Some("-V") => {
            println!("qwen38-metal {}", env!("CARGO_PKG_VERSION"));
        }
        Some(command) => {
            eprintln!("unknown command: {command}");
            eprintln!("run `qwen38-metal help` for usage");
            std::process::exit(2);
        }
    }
}

fn print_help() {
    println!(
        "qwen38-metal\n\n\
         Native Rust and precompiled Metal substrate for Qwen3.8 inference.\n\n\
         Usage:\n\
           qwen38-metal doctor\n\
           qwen38-metal version\n\n\
         The current milestone validates the binary and embedded Metal library.\n\
         Model loading and inference are introduced in later milestones."
    );
}

fn print_doctor() {
    println!("binary: qwen38-metal {}", env!("CARGO_PKG_VERSION"));
    println!("runtime: native Rust");
    println!("Metal source: precompiled at build time");
    println!("embedded metallib: {} bytes", METALLIB.len());
    println!("target profile: aarch64-apple-darwin");
}
