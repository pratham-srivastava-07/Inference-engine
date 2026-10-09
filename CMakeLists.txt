cmake_minimum_required(VERSION 3.20)

project(mini_infer
    VERSION 0.1.0
    DESCRIPTION "A small transformer inference runtime, built from scratch for learning"
    LANGUAGES CXX
)

# Make our helper modules in cmake/ findable by include().
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

option(MINI_INFER_BUILD_TESTS "Build the mini-infer test suite" ON)

# ---------------------------------------------------------------------------
# Core library: everything reusable lives here. The app and tests link to it.
# ---------------------------------------------------------------------------
add_library(mini_infer STATIC
    src/tensor.cpp
    src/linear.cpp
    src/relu.cpp
)

target_include_directories(mini_infer
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)

# PUBLIC: anything that links mini_infer is also compiled as C++20.
target_compile_features(mini_infer PUBLIC cxx_std_20)
mini_infer_set_project_options(mini_infer)

# ---------------------------------------------------------------------------
# Executable: a tiny program that proves the library builds and links.
# ---------------------------------------------------------------------------
add_executable(mini_infer_cli
    src/main.cpp
)

target_link_libraries(mini_infer_cli PRIVATE mini_infer)
mini_infer_set_project_options(mini_infer_cli)

# ---------------------------------------------------------------------------
# Tests (CTest). enable_testing() must be called from the top-level
# CMakeLists.txt so that `ctest` can be run from the build directory root.
# ---------------------------------------------------------------------------
if(MINI_INFER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
