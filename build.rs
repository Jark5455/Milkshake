fn main() {
    cc::Build::new()
        .cpp(true)
        .include("/usr/include/torch/csrc/api/include/")
        .file("src/cpp/cmaes.cpp")
        .file("src/cpp/cmaes.h")
        .compile("cmaes");
}
