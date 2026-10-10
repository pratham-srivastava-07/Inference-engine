#undef NDEBUG
#include <cassert>
#include<iostream>

#include "mini_infer/layer_normalization.hpp"
#include "mini_infer/tensor.hpp"

using mini_infer::LayerNormalization;
using mini_infer::Tensor;

void testLayerNorm() {
    Tensor<float> input({3});

    input[0] = 2.0f;
    input[1] = 4.0f;
    input[2] = 6.0f;

    LayerNormalization<float> layerNorm;
    Tensor<float> output = layerNorm.forward(input);

    constexpr float eps = 1e-4f;

    assert(std::abs(output[0] + 1.2247f) < eps);
    assert(std::abs(output[1] - 0.0f) < eps);
    assert(std::abs(output[2] - 1.2247f) < eps);
}


int main() {
    testLayerNorm();
    return 0;
}