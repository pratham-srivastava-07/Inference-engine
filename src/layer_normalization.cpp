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

template<typename T>
Tensor<T> LayerNormalization<T>::forward_two(const Tensor<T>& input) const {

    if (input.shape().size() != 2 || input.size() == 0) {
        throw std::invalid_argument(
            "2D normalization requires a non empty array"
        );
    }

    if (input.shape()[1] != gamma_.size()) {
        throw std::invalid_argument(
            "Input feature count must match gamma and beta"
        );
    }

    std::size_t rows = input.shape()[0];
    std::size_t cols = input.shape()[1];

    Tensor<T> output(input.shape());

    for (std::size_t i = 0; i < rows; i++) {
        T sum = T{};        
        // mean
        for (std::size_t j = 0; j < cols; j++) {
            sum += input(i, j);
        }
        T mean = sum / static_cast<T>(cols);

        // variance 
        T squared_diff = T{};

        for (std::size_t j = 0; j < cols; j++) {
            squared_diff += std::pow(input(i, j)-mean, T{2});
        }

        T variance = squared_diff / static_cast<T>(cols);

        // normalization of 2d
        T epsilon = static_cast<T>(1e-5);

        T denominator = std::sqrt(variance + epsilon);

        for (std::size_t j = 0; j < cols; j++) {
            T normalized = (input(i,j) - mean) / denominator;
            output(i, j) = (gamma_[j] * normalized) + beta_[j]; 
        }
    }

    return output;
}

template<typename T>
LayerNormalization<T>::LayerNormalization(const Tensor<T>& gamma,
 const Tensor<T>& beta) : gamma_(gamma), beta_(beta) {
    // validate the data members while construction

    if (gamma_.size() == 0 || beta_.size() == 0) 
    {
        throw std::invalid_argument(
            "Gamma and beta must not be empty"
        );
    }

    if (gamma_.shape().size() != 1 || beta_.shape().size() != 1) 
    {
        throw std::invalid_argument(
            "Gamma and beta both should be 1 dimentional"
        );
    }

    if (gamma_.size() != beta_.size()) 
    {
        throw std::invalid_argument(
            "Gamma and beta should have same number of elements"
        );
    }

 }

template class LayerNormalization<float>;

}  // namespace mini_infer
