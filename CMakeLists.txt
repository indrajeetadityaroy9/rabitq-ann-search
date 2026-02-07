cmake_minimum_required(VERSION 3.18)
project(cp-hnsw VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(CPHNSW_BUILD_TESTS "Build unit tests" ON)
option(CPHNSW_BUILD_EVAL "Build evaluation framework" ON)
option(CPHNSW_BUILD_PYTHON "Build Python bindings (requires pybind11)" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -O3)

    # SIMD flags - Auto-detect or assume SOTA hardware
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        add_compile_options(-mavx512f -mavx512bw -mavx512vl)
    else()
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            add_compile_options(-mavx2)
        endif()
    endif()

    # Enable native optimizations
    if(NOT APPLE)
        add_compile_options(-march=native)
    endif()

elseif(MSVC)
    add_compile_options(/W4 /O2)
    add_compile_options(/arch:AVX512)
endif()

# OpenMP support (Required)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(FATAL_ERROR "OpenMP is required for SOTA performance.")
endif()

# Header-only library target
add_library(cphnsw INTERFACE)
target_include_directories(cphnsw INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/cpp>
    $<INSTALL_INTERFACE:include>
)

# Link OpenMP
target_link_libraries(cphnsw INTERFACE OpenMP::OpenMP_CXX)

# Evaluation executable
if(CPHNSW_BUILD_EVAL)
    add_executable(eval_master src/cpp/eval_master.cpp)
    target_link_libraries(eval_master PRIVATE cphnsw)

    # Dimension-specific eval_master variants
    foreach(D_VAL 128 256 512)
        add_executable(eval_master_d${D_VAL} src/cpp/eval_master.cpp)
        target_compile_definitions(eval_master_d${D_VAL} PRIVATE EVAL_D=${D_VAL})
        target_link_libraries(eval_master_d${D_VAL} PRIVATE cphnsw)
    endforeach()
endif()

# Unit tests
if(CPHNSW_BUILD_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/test_rabitq.cpp")
        add_executable(test_rabitq tests/cpp/test_rabitq.cpp)
        target_link_libraries(test_rabitq PRIVATE cphnsw)
    endif()
endif()

# Python bindings
if(CPHNSW_BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_core src/cpp/bindings.cpp)
    target_link_libraries(_core PRIVATE cphnsw)
endif()

# Install
install(TARGETS cphnsw EXPORT cphnsw-targets)
install(DIRECTORY src/cpp/cphnsw/ DESTINATION include/cphnsw)

# Print configuration summary
message(STATUS "")
message(STATUS "CP-HNSW Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Tests: ${CPHNSW_BUILD_TESTS}")
message(STATUS "  Build Evaluation: ${CPHNSW_BUILD_EVAL}")
message(STATUS "  Build Python: ${CPHNSW_BUILD_PYTHON}")
message(STATUS "")
