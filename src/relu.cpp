#include <bits/stdc++.h>

#include "mini_infer/relu.hpp"

using namespace std;

namespace mini_infer {
    template<typename T>
    Tensor<T> Relu<T>::forward(const Tensor<T> input) const {
        Tensor<T> output(input.shape());

        for (std::size_t i=0; i<input.size(); i++) {
            if (input[i] < T{}) {
                output[i] = T{}; // type generic way of representing 0
            }


            else {
                output[i] = input[i];
            }
        }
        retur output;
    }
}