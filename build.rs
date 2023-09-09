fn main() {
    let main_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("{}", main_dir);

    let command = std::process::Command::new("nvcc")
        .arg("-ptx")
        .arg(format!("{}/cuda/loss.cu", main_dir))
        .arg("-o")
        .arg(format!("{}/target/loss.ptx", main_dir));

    println!("{:?}", command);

    command.spawn().expect("Failed to compile kernel");
}