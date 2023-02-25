cmake_minimum_required(VERSION 3.24)
project(Milkshake LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Milkshake src/main.cu src/buffer.cu src/stockframe.cu)

find_package(Arrow REQUIRED)
find_package(OpenSSL REQUIRED)

target_link_libraries(Milkshake Arrow::arrow_shared cublas cublasLt cudnn ta_lib z)
set_target_properties(Milkshake PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
