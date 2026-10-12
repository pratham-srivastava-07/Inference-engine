#undef NDEBUG
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "mini_infer/gelu.hpp"

using mini_infer::Tensor;
using mini_infer::Gelu;

void test_gelu_values() {
    Tensor<float> input({5});
    input[0] = -2.0f;
    input[1] = -1.0f;
    input[2] =  0.0f;
    input[3] =  1.0f;
    input[4] =  2.0f;

    Gelu<float> gelu;
    auto output = gelu.forward(input);

    const float expected[] = {
        -0.0454023f,
        -0.158808f,
         0.0f,
         0.841192f,
         1.954598f
    };

    for (std::size_t i = 0; i < input.size(); ++i) {
        assert(std::abs(output[i] - expected[i]) < 1e-4f);
    }
}

void test_gelu_preserves_shape() {
    Tensor<float> input({2, 3});

    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) - 2.0f;
    }

    Gelu<float> gelu;
    auto output = gelu.forward(input);

    assert(output.shape() == input.shape());
}

void test_gelu_does_not_modify_input() {
    Tensor<float> input({3});
    input[0] = -1.0f;
    input[1] = 0.5f;
    input[2] = 2.0f;

    const float original[] = {-1.0f, 0.5f, 2.0f};

    Gelu<float> gelu;
    auto output = gelu.forward(input);

    for (std::size_t i = 0; i < input.size(); ++i) {
        assert(input[i] == original[i]);
    }
}

void test_gelu_2d_values() {
    Tensor<float> input({2, 2});
    input[0] = -1.0f;
    input[1] =  0.0f;
    input[2] =  1.0f;
    input[3] =  2.0f;

    Gelu<float> gelu;
    auto output = gelu.forward(input);

    assert(std::abs(output[0] - (-0.158808f)) < 1e-4f);
    assert(std::abs(output[1] - 0.0f) < 1e-4f);
    assert(std::abs(output[2] - 0.841192f) < 1e-4f);
    assert(std::abs(output[3] - 1.954598f) < 1e-4f);
}

void test_gelu_empty_input() {
    // Only include this test if your Tensor implementation permits
    // construction of a zero-sized tensor.
    Tensor<float> input({0});

    Gelu<float> gelu;
    bool threw = false;

    try {
        (void)gelu.forward(input);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

int main() {
    test_gelu_values();
    test_gelu_preserves_shape();
    test_gelu_does_not_modify_input();
    test_gelu_2d_values();
    test_gelu_empty_input();

    std::cout << "All GELU tests passed!\n";
    return 0;
}