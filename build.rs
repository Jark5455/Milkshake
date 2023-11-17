fn main() {
    println!("cargo:rustc-link-lib=mujoco");
    println!("cargo:rerun-if-changed=src/mujoco/wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("src/mujoco/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_default(true)
        .blocklist_item("FP_NAN")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_NORMAL")

        .generate()
        .expect("Unable to generate bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("mujoco-bindings.rs"))
        .expect("Couldn't write bindings!");
}