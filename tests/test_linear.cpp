#undef NDEBUG
#include <cassert>
#include<iostream>

#include "mini_infer/linear.hpp"

using mini_infer::Tensor;
using mini_infer::Linear;

void test_linear_layer() {
    Tensor<float> weights({2, 2});
    Tensor<float> bias({2});

    weights(0, 0) = 1;
    weights(0, 1) = 2;
    weights(1, 0) = 3;
    weights(1, 1) = 4;

    bias({0}) = 10;
    bias({1}) = 20;

  

    Linear<float> layer(2, 2, weights, bias);

    Tensor<float> input({1,2});

    input(0,0) = 2;
    input(0,1) = 3;

    Tensor<float> output = layer.forward(input);

    assert(output.shape() == (std::vector<std::size_t>{1, 2}));
    assert(output(0,0) == 21);
    assert(output(0,1) == 36);

}

int main() {
    test_linear_layer();
    return 0;
}

// XW = [11, 16], + bias [10, 20] = [21, 36]