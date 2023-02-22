cmake_minimum_required(VERSION 3.24)
project(Milkshake LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_library(pch src/pch.cpp)
target_precompile_headers(pch PUBLIC src/pch.h)

add_executable(Milkshake src/main.cu src/buffer.cu src/stockframe.cu)
target_precompile_headers(Milkshake REUSE_FROM pch)

find_package(Arrow REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(Milkshake ${Boost_LIBRARIES})
endif()

target_link_libraries(Milkshake Arrow::arrow_shared cublas cublasLt cudnn ta_lib z)
set_target_properties(Milkshake PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
