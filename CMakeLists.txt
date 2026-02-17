cmake_minimum_required(VERSION 3.18)
project(cp-hnsw VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -O3 -march=native -mavx2 -mfma)

find_package(OpenMP REQUIRED)

add_library(cphnsw INTERFACE)
target_include_directories(cphnsw INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cphnsw INTERFACE OpenMP::OpenMP_CXX)

find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(_core src/bindings.cpp)
    target_link_libraries(_core PRIVATE cphnsw)
endif()

install(TARGETS cphnsw EXPORT cphnsw-targets)
install(DIRECTORY include/cphnsw/ DESTINATION include/cphnsw)
