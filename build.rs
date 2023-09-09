fn main() {
    let main_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    std::process::Command::new("nvcc")
        .arg("-ptx")
        .arg(format!("{}/cuda/loss.cu", main_dir))
        .arg("-o")
        .arg(format!("{}/target/loss.ptx", main_dir))
        .spawn().expect("Failed to compile kernel");
}