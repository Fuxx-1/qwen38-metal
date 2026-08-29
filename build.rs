use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders/qwen38.metal");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let source = PathBuf::from("shaders/qwen38.metal");
    let air = out_dir.join("qwen38.air");
    let metallib = out_dir.join("qwen38.metallib");

    run_xcrun(
        Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg(&source)
            .arg("-o")
            .arg(&air),
        "compile Metal source",
    );
    run_xcrun(
        Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib"])
            .arg(&air)
            .arg("-o")
            .arg(&metallib),
        "link Metal library",
    );

    println!("cargo:rustc-env=QWEN38_METALLIB={}", metallib.display());
}

fn run_xcrun(command: &mut Command, step: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to {step}: {error}"));

    if !status.success() {
        panic!("failed to {step}: xcrun exited with {status}");
    }
}
