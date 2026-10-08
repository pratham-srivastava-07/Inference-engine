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


template<typename T>
void Tensor<T>::computeStrides() {
    strides.resize(shape_.size());

    if (strides.empty()) return;

    strides.back() = 1;

    // run backwards and calculate strides 

    for (std::size_t i = strides.size()-1; i > 0; i--) {
        strides[i-1] = strides[i] * shape_[i];
    }
     // a dimention's stride is equal to the product of all shape dimentions to its right

    return;
}


template<typename T>
T& Tensor<T>::operator()(const std::vector<std::size_t>& coordinates) {
    
    if (coordinates.size() != shape_.size()) {
        throw invalid_argument("Invalid shape of coordinates vector");
    }

    std::size_t flattened_index = 0;

    for (std::size_t i = 0; i < coordinates.size(); i++) {
        if (coordinates[i] >= shape_[i]) {
            flattened_index += strides[i] * coordinates[i];
        }
    }

    return data_[flattened_index];
}

template<typename T>
const T& Tensor<T>::operator()(const std::vector<std::size_t>& coordinates) const {
    if (coordinates.size() != shape_.size()) {
        throw invalid_argument("Invalid shape of coordinates vector");
    }

    std::size_t flattened_index = 0;

    for (std::size_t i = 0; i < coordinates.size(); i++) {
        if (coordinates[i] >= shape_[i]) {
            flattened_index += strides[i] * coordinates[i];
        }
    }

    return data_[flattened_index];
}


template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& vec) const {
    Tensor<T> result(shape_);

    if (shape_ != vec.shape_) {
        throw invalid_argument("Tensor shape must match");
    }

    for (std::size_t i = 0; i <= shape_.size()-1; i++) {
        result.data_[i] = data_[i] + vec.data_[i];
    }

    return result;
}