cmake_minimum_required(VERSION 3.18)
project(cp-hnsw VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(CPHNSW_BUILD_TESTS "Build unit tests" ON)
option(CPHNSW_BUILD_EVAL "Build evaluation framework" ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -O3)

    # SIMD flags - Auto-detect or assume SOTA hardware
    # We add flags to enable AVX-512 if compiler supports it
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
    # Assume modern hardware
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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link OpenMP
target_link_libraries(cphnsw INTERFACE OpenMP::OpenMP_CXX)

# Evaluation executable
if(CPHNSW_BUILD_EVAL)
    # Master evaluation protocol (PhD portfolio)
    add_executable(eval_master
        evaluation/eval_master.cpp
    )
    target_link_libraries(eval_master PRIVATE cphnsw)
    target_include_directories(eval_master PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
    )

    # K-specific eval_master variants
    foreach(K_VAL 16 32 64)
        add_executable(eval_master_k${K_VAL}
            evaluation/eval_master.cpp
        )
        target_compile_definitions(eval_master_k${K_VAL} PRIVATE EVAL_K=${K_VAL})
        target_link_libraries(eval_master_k${K_VAL} PRIVATE cphnsw)
        target_include_directories(eval_master_k${K_VAL} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
        )
    endforeach()
endif()

# Unit tests
if(CPHNSW_BUILD_TESTS)
    # Try to find GTest
    find_package(GTest QUIET)

    if(GTest_FOUND)
        enable_testing()

        add_executable(cphnsw_tests
            tests/unit/test_hadamard.cpp
        )
        target_link_libraries(cphnsw_tests PRIVATE cphnsw GTest::gtest_main)

        include(GoogleTest)
        gtest_discover_tests(cphnsw_tests)
    else()
        message(STATUS "GTest not found, skipping tests")
    endif()

    # Standalone test executables
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_unified_distance.cpp")
        add_executable(test_unified_distance
            tests/unit/test_unified_distance.cpp
        )
        target_link_libraries(test_unified_distance PRIVATE cphnsw)
        target_include_directories(test_unified_distance PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
    endif()
endif()

# Install
install(TARGETS cphnsw EXPORT cphnsw-targets)
install(DIRECTORY include/ DESTINATION include)

# Print configuration summary
message(STATUS "")
message(STATUS "CP-HNSW Configuration (SOTA):")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Tests: ${CPHNSW_BUILD_TESTS}")
message(STATUS "  Build Evaluation: ${CPHNSW_BUILD_EVAL}")
message(STATUS "")
