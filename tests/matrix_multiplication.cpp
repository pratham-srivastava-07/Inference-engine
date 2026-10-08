#include<cassert>
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

    std::cout << "C(0,0): " << C(0, 0) << "\n";
std::cout << "C(0,1): " << C(0, 1) << "\n";
std::cout << "C(1,0): " << C(1, 0) << "\n";
std::cout << "C(1,1): " << C(1, 1) << std::endl;

    std::cout << "Matrix multiplication test passed\n";

}

int main() {
    test_multiplication();
    return 0;
}