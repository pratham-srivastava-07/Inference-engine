#include <bits/stdc++.h>
using namespace std;

#include <stdexcept>
#include <cmath>
#include <vector>

#include "mini_infer/layer_normalization.hpp"

namespace mini_infer {

template<typename T>
Tensor<T> LayerNormalization<T>::forward(const Tensor<T>& input) const {

    if (input.shape().size() != 1 || input.size() == 0) {
        throw std::invalid_argument("1d normalization requires a non empty array");
    }


    T sum = T{};

    for (std::size_t i = 0; i < input.size(); i++) {
        sum += input[i];
    }

    T mean = sum / static_cast<T>(input.size());
    
    // calculate variance 
    T squared_diff = T{};
    for (std::size_t i = 0; i < input.size(); i++) {
        squared_diff += std::pow(input[i]-mean, T{2});
    }

    T variance = squared_diff / static_cast<T>(input.size());

    // normalization part
    T epsilon = static_cast<T>(1e-5);

    T denominator = std::sqrt(variance + epsilon);

    Tensor<T> output(input.shape());

    for (std::size_t i = 0; i < input.size(); i++){
        output[i] = (input[i] - mean) / denominator;
    }

    return output;

}    
template class LayerNormalization<float>;
}