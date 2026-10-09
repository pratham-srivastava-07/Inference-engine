#undef NDEBUG
#include <cassert>
#include<iostream>

#include "mini_infer/relu.hpp"


using mini_infer::Tensor;
using mini_infer::Relu;

void test_relu() {
    Tensor<float> input({2, 3});

    input[0] = -5;
    input[1] = 3;
    input[2] = -2;
    input[3] = 7;
    input[4] = 0;
    input[5] = -1;

    Relu<float> relu;
    Tensor<float> output = relu.forward(input);

    assert(output.shape() == (std::vector<std::size_t>{2, 3}));

    // negatives -> 0, everything else unchanged
    assert(output[0] == 0);
    assert(output[1] == 3);
    assert(output[2] == 0);
    assert(output[3] == 7);
    assert(output[4] == 0);
    assert(output[5] == 0);
}

int main() {
    test_relu();
    return 0;
}
