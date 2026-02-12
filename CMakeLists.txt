cmake_minimum_required(VERSION 3.18)
project(cp-hnsw VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags - SOTA Hardware assumed
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -O3)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        add_compile_options(-mavx512f -mavx512bw -mavx512vl -mavx2)
        if(NOT APPLE)
            add_compile_options(-march=native)
        endif()
    endif()

elseif(MSVC)
    add_compile_options(/W4 /O2)
    add_compile_options(/arch:AVX512)
endif()

# OpenMP support
find_package(OpenMP REQUIRED)

# Header-only library target
add_library(cphnsw INTERFACE)
target_include_directories(cphnsw INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cphnsw INTERFACE OpenMP::OpenMP_CXX)

# Evaluation executable
add_executable(benchmark apps/benchmark.cpp)
target_link_libraries(benchmark PRIVATE cphnsw)

foreach(D_VAL 128 256 512)
    add_executable(benchmark_d${D_VAL} apps/benchmark.cpp)
    target_compile_definitions(benchmark_d${D_VAL} PRIVATE EVAL_D=${D_VAL})
    target_link_libraries(benchmark_d${D_VAL} PRIVATE cphnsw)
endforeach()

# Multi-bit test
add_executable(test_nbit apps/test_nbit.cpp)
target_link_libraries(test_nbit PRIVATE cphnsw)

# ITQ test
add_executable(test_itq apps/test_itq.cpp)
target_link_libraries(test_itq PRIVATE cphnsw)

# HNSW test
add_executable(test_hnsw apps/test_hnsw.cpp)
target_link_libraries(test_hnsw PRIVATE cphnsw)

# Python bindings
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(_core src/bindings.cpp)
    target_link_libraries(_core PRIVATE cphnsw)
endif()

# Install
install(TARGETS cphnsw EXPORT cphnsw-targets)
install(DIRECTORY include/cphnsw/ DESTINATION include/cphnsw)

message(STATUS "CP-HNSW SOTA Configuration Active")