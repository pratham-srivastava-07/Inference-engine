// Keep assert() active even in Release builds (which define NDEBUG),
// otherwise the checks below would silently compile away.
#undef NDEBUG
#include <cassert>
#include<iostream>

#include "mini_infer/tensor.hpp"

using mini_infer::Tensor;

void test_multiplication() {
    Tensor<int> A({2, 3});
    Tensor<int> B({3, 2});

    // Initialize A
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    // Initialize B
    B(0, 0) = 7;
    B(0, 1) = 8;
    B(1, 0) = 9;
    B(1, 1) = 10;
    B(2, 0) = 11;
    B(2, 1) = 12;

    Tensor<int> C = A * B;

    // [1 2 3]   [ 7  8]   [ 58  64]
    // [4 5 6] x [ 9 10] = [139 154]
    //           [11 12]
    assert(C(0, 0) == 58);
    assert(C(0, 1) == 64);
    assert(C(1, 0) == 139);
    assert(C(1, 1) == 154);

    std::cout << "Matrix multiplication test passed\n";

}

int main() {
    test_multiplication();
    return 0;
}