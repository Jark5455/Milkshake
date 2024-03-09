fn main() {
    println!("cargo:rustc-link-lib=mujoco");
    println!("cargo:rerun-if-changed=src/wrappers/mujoco_wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("src/wrappers/mujoco_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_default(true)
        .allowlist_function("mj.*")
        .allowlist_type("mj.*")
        .allowlist_var("mj.*")
        .generate()
        .expect("Unable to generate mujoco bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("mujoco-bindings.rs"))
        .expect("Couldn't write mujoco bindings!");

    println!("cargo:rustc-link-lib=ta_lib");
    println!("cargo:rerun-if-changed=src/wrappers/talib_wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("src/wrappers/talib_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_default(true)
        .allowlist_function("TA.*")
        .allowlist_type("TA.*")
        .allowlist_var("TA.*")
        .generate()
        .expect("Unable to generate talib bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("talib-bindings.rs"))
        .expect("Couldn't write talib bindings!");
}
