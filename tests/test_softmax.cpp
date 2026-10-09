
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "../src/softmax.cpp"

using mini_infer::Tensor;
using mini_infer::Softmax;

constexpr float EPS = 1e-4f;

// Test 1: 1D Softmax
void test_softmax_1d() {
    Tensor<float> input({3});

    input[0] = 1.0f;
    input[1] = 2.0f;
    input[2] = 3.0f;

    Softmax<float> softmax;
    Tensor<float> output = softmax.forward(input);

    // Check output shape
    assert((output.shape() == std::vector<std::size_t>{3}));

    // Check probabilities
    assert(std::abs(output[0] - 0.0900306f) < EPS);
    assert(std::abs(output[1] - 0.2447285f) < EPS);
    assert(std::abs(output[2] - 0.6652409f) < EPS);

    // Probabilities must sum to 1
    float sum = 0.0f;

    for (std::size_t i = 0; i < output.size(); ++i) {
        sum += output[i];
    }

    assert(std::abs(sum - 1.0f) < EPS);

    // Original input must remain unchanged
    assert(input[0] == 1.0f);
    assert(input[1] == 2.0f);
    assert(input[2] == 3.0f);

    std::cout << "1D Softmax test passed!\n";
}

// Test 2: Row-wise 2D Softmax
void test_softmax_2d() {
    Tensor<float> input({2, 3});

    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    input(1, 0) = 4.0f;
    input(1, 1) = 5.0f;
    input(1, 2) = 6.0f;

    Softmax<float> softmax;
    Tensor<float> output = softmax.forward_two(input);

    // Check output shape
    assert((output.shape() == std::vector<std::size_t>{2, 3}));

    // Expected probabilities for both rows
    constexpr float expected[] = {
        0.0900306f,
        0.2447285f,
        0.6652409f
    };

    // Validate every element and each row's sum
    for (std::size_t i = 0; i < 2; ++i) {
        float row_sum = 0.0f;

        for (std::size_t j = 0; j < 3; ++j) {
            assert(std::abs(output(i, j) - expected[j]) < EPS);
            row_sum += output(i, j);
        }

        assert(std::abs(row_sum - 1.0f) < EPS);
    }

    // Original input must remain unchanged
    assert(input(0, 0) == 1.0f);
    assert(input(0, 1) == 2.0f);
    assert(input(0, 2) == 3.0f);
    assert(input(1, 0) == 4.0f);
    assert(input(1, 1) == 5.0f);
    assert(input(1, 2) == 6.0f);

    std::cout << "2D Softmax test passed!\n";
}

int main() {
    test_softmax_1d();
    test_softmax_2d();

    std::cout << "All Softmax tests passed!\n";
    return 0;
}
