# mini-infer

A small transformer inference runtime written from scratch in C++20, for learning.

**Status:** project scaffold only. There are no tensor operations, layers,
tokenizer, or model loading yet.

## Layout

```
CMakeLists.txt            top-level build: library, executable, tests
cmake/CompilerWarnings.cmake   per-target warning/standard settings
include/mini_infer/       public headers (tensor.hpp)
src/                      library sources + main.cpp for the CLI
tests/                    CTest-driven test executables
```

## Requirements

- CMake 3.20+
- A C++20 compiler (GCC 11+, Clang 14+, or MSVC 19.30+)
- Optional: Ninja

## Build and run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
./build/mini_infer_cli
```

On Windows the executable is `build\mini_infer_cli.exe`.

### Windows + MinGW note

If the executable crashes immediately when launched from **Git Bash** but works
from PowerShell/cmd, Git Bash is putting its own (older) `libstdc++-6.dll`
ahead of your compiler's on `PATH`. Run from PowerShell, or put your MinGW
`bin` directory first on `PATH`.
