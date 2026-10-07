#include "mini_infer/tensor.hpp"
#include<bits/stdc++.h>

using namespace std;

// defining and implementing member initializer list for tensor class
template<typename T>
Tensor::Tensor<T>(const std::vector<std::size_t>& shape) : shape_(shape) {
    std::size_t totalSize = 1;

    for (auto dimension : shape) {
        totalSize *= dimension;
    }

    data_.resize(totalSize);
}


template<typename T>
T& Tensor::operator()(std::size_t row, std::size_t col) {
    // if (shape_.size() == 2  && (row < shape_[0] && col < shape_[1])) {
    //     int flattened_index = row * shape_[1] + col;

    //     return data_[flattened_index];
    // }

    if (shape_.size() != 2) {
        throw std::invalid_argument("Tensor must be 2d");
    }

    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::invalid_argument("Row and columns must be inbound");
    }

    std::size_t flattened_index = row * shape_[1] + col;

    return data_[flattened_index];
}

template<typename T>
const T& Tensor::operator()(std::size_t row, std::size_t col) const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Tensor must be 2d");
    }

    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::invalid_argument("Row and columns must be inbound");
    }

    std::size_t flattened_index = row * shape_[1] + col;

    return data_[flattened_index];
}


