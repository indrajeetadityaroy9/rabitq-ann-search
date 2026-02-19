cmake_minimum_required(VERSION 3.18)
project(cp-hnsw VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -O3)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    message(FATAL_ERROR
        "CP-HNSW requires x86_64 architecture. "
        "Detected: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
if(NOT COMPILER_SUPPORTS_AVX2 OR NOT COMPILER_SUPPORTS_FMA)
    message(FATAL_ERROR
        "CP-HNSW requires AVX2 and FMA support. "
        "AVX2=${COMPILER_SUPPORTS_AVX2} FMA=${COMPILER_SUPPORTS_FMA}")
endif()
add_compile_options(-march=native -mavx2 -mfma)

find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

add_library(cphnsw INTERFACE)
target_include_directories(cphnsw INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cphnsw INTERFACE OpenMP::OpenMP_CXX)

pybind11_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE cphnsw)

install(TARGETS cphnsw EXPORT cphnsw-targets)
install(DIRECTORY include/cphnsw/ DESTINATION include/cphnsw)
