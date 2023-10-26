extern crate anyhow;
extern crate cc;
extern crate glob;

const PYTHON_PRINT_PYTORCH_DETAILS: &str = "
import torch \n
from torch.utils import cpp_extension \n
print('LIBTORCH_CXX11:', torch._C._GLIBCXX_USE_CXX11_ABI) \n
for include_path in cpp_extension.include_paths(): \n
  print('LIBTORCH_INCLUDE:', include_path) \n
for library_path in cpp_extension.library_paths(): \n
  print('LIBTORCH_LIB:', library_path)
";

enum Os {
    Linux,
    Macos,
    Windows,
}

fn main() -> anyhow::Result<()> {
    let srcfiles = glob::glob("src/cpp/*.cpp")?.collect::<Result<Vec<std::path::PathBuf>, glob::GlobError>>()?;

    for file in srcfiles.clone() {
        println!("cargo:rerun-if-changed={}", file.into_os_string().into_string().expect("Invalid UTF-8 Path"));
    }

    let os = match std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS").as_str() {
        "linux" => Os::Linux,
        "windows" => Os::Windows,
        "macos" => Os::Macos,
        os => anyhow::bail!("unsupported TARGET_OS '{os}'"),
    };

    let python_interpreter = match os {
        Os::Windows => std::path::PathBuf::from("python.exe"),
        Os::Linux | Os::Macos => {
            if std::env::var_os("VIRTUAL_ENV").is_some() {
                std::path::PathBuf::from("python")
            } else {
                std::path::PathBuf::from("python3")
            }
        }
    };

    let output = std::process::Command::new(&python_interpreter)
        .arg("-c")
        .arg(PYTHON_PRINT_PYTORCH_DETAILS)
        .output()?;

    if !output.status.success() {
        anyhow::bail!("Unable to query pytorch information from python: {}", String::from_utf8_lossy(output.stderr.as_slice()));
    }

    let mut libtorch_include_dirs = vec![];
    let mut libtorch_lib_dirs = vec![];
    let mut cxx11_abi = None;

    for line in String::from_utf8_lossy(output.stdout.as_slice()).lines() {
        if let Some(path) = line.strip_prefix("LIBTORCH_INCLUDE: ") {
            libtorch_include_dirs.push(std::path::PathBuf::from(path))
        }

        if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
            libtorch_lib_dirs.push(std::path::PathBuf::from(path))
        }

        match line.strip_prefix("LIBTORCH_CXX11: ") {
            Some("True") => cxx11_abi = Some("1"),
            Some("False") => cxx11_abi = Some("0"),
            _ => ()
        }
    }

    if cxx11_abi.is_none() {
        anyhow::bail!("No pytorch cxx abi information found")
    }

    if libtorch_include_dirs.len() == 0 {
        anyhow::bail!("No pytorch include directories found");
    }

    if libtorch_lib_dirs.len() == 0 {
        anyhow::bail!("No pytorch library directories found");
    }

    let mut build = cc::Build::new();
    build.std("c++17");
    build.flag(format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11_abi.unwrap()).as_str());

    for file in srcfiles {
        build.file(file);
    }

    build.includes(libtorch_include_dirs);

    for lib in libtorch_lib_dirs {
        println!("cargo:rustc-link-search=native={}", lib.into_os_string().into_string().unwrap());
    }

    println!("cargo:rustc-link-lib=dylib=torch_cuda");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=c10_cuda");
    println!("cargo:rustc-link-lib=dylib=c10");

    build.try_compile("pytorch-milkshake")?;

    anyhow::Ok(())
}
