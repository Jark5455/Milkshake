extern crate anyhow;
extern crate cc;
extern crate glob;
extern crate torch_build;

fn main() -> anyhow::Result<()> {
    let srcfiles = glob::glob("src/cpp/*.cpp")?.collect::<Result<Vec<std::path::PathBuf>, glob::GlobError>>()?;
    let mut cargo_commands = vec![];

    let mut build = cc::Build::new();

    torch_build::build_cpp(
        &mut build,
        true,
        false,
        Some(&mut cargo_commands),
        srcfiles.clone()
    )?;

    build.try_compile("pytorch-milkshake")?;

    for file in srcfiles {
        println!("cargo:rerun-if-changed={:?}", file.into_os_string().into_string().expect("Invalid UTF-8 Path"));
    }

    cargo_commands.iter().for_each(|command| {
        println!("{}", command);
    });

    anyhow::Ok(())
}