#undef NDEBUG
#include <cassert>
#include <type_traits>

#include "mini_infer/tensor.hpp"

using mini_infer::Tensor;

// Compile-time checks: if these fail, the test does not even build.
static_assert(std::is_default_constructible_v<mini_infer::Tensor<float>>);
static_assert(std::is_default_constructible_v<mini_infer::Tensor<int>>);

void test_transpose() {
    Tensor<int> input({2, 3});
    input(0, 0) = 1;
    input(0, 1) = 2;
    input(0, 2) = 3;
    input(1, 0) = 4;
    input(1, 1) = 5;
    input(1, 2) = 6;

    auto output = input.transpose();

    assert(output.shape()[0] == 3);
    assert(output.shape()[1] == 2);

    assert(output(0, 0) == 1);
    assert(output(0, 1) == 4);
    assert(output(1, 0) == 2);
    assert(output(1, 1) == 5);
    assert(output(2, 0) == 3);
    assert(output(2, 1) == 6);

    // Original tensor must remain unchanged.
    assert(input(0, 1) == 2);
    assert(input(1, 0) == 4);
}

int main() {
    [[maybe_unused]] mini_infer::Tensor<float> tensor;
    test_transpose();
    return 0;  // 0 tells CTest the test passed
}
