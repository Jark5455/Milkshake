extern crate anyhow;
extern crate cc;
extern crate glob;

const PYTHON_PRINT_PYTORCH_DETAILS: &str = "
from torch.utils import cpp_extension \n
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

    assert!(output.status.success());

    let mut libtorch_include_dirs = vec![];
    let mut libtorch_lib_dirs = vec![];

    for line in String::from_utf8(output.stdout)?.lines() {
        if let Some(path) = line.strip_prefix("LIBTORCH_INCLUDE: ") {
            libtorch_include_dirs.push(std::path::PathBuf::from(path))
        }

        if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
            libtorch_lib_dirs.push(std::path::PathBuf::from(path))
        }
    }

    assert_ne!(libtorch_include_dirs.len(), 0);
    assert_ne!(libtorch_lib_dirs.len(), 0);

    let mut build = cc::Build::new();

    for file in srcfiles {
        build.file(file);
    }

    for include in libtorch_include_dirs {
        build.include(include.into_os_string().into_string().unwrap().as_str());
    }

    for lib in libtorch_lib_dirs {
        build.flag(format!("-L {}", lib.into_os_string().into_string().unwrap()).as_str());
    }

    build.flag("-ltorch_cuda");
    build.flag("-ltorch_cpu");
    build.flag("-ltorch");
    build.flag("-lc10");

    build.try_compile("pytorch-milkshake")?;

    anyhow::Ok(())
}