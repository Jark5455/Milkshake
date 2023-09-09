fn main() {
    std::process::Command::new("nvcc")
        .arg("-ptx")
        .arg("cuda/loss.cu")
        .arg("-o")
        .arg(format!("{}/target/loss.ptx", std::env::var("CARGO_MANIFEST_DIR").unwrap()))
        .spawn().expect("Failed to compile kernel");
}