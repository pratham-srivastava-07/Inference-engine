#include <type_traits>

#include "mini_infer/tensor.hpp"

// Compile-time checks: if these fail, the test does not even build.
static_assert(std::is_default_constructible_v<mini_infer::Tensor<float>>);
static_assert(std::is_default_constructible_v<mini_infer::Tensor<int>>);

int main() {
    [[maybe_unused]] mini_infer::Tensor<float> tensor;
    return 0;  // 0 tells CTest the test passed
}
