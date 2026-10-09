
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "mini_infer/tensor.hpp"

// using namespace std;

namespace mini_infer {

template <typename T>
class Softmax {
    static_assert(
        std::is_floating_point_v<T>,
        "Softmax requires a floating-point type"
    );

public:
    Tensor<T> forward(const Tensor<T>& input) const {
        // For now, support only nonempty 1D tensors
        if (input.shape().size() != 1 || input.size() == 0) {
            throw std::invalid_argument(
                "Softmax requires a nonempty 1D tensor"
            );
        }

        // Step 1: Find maximum for numerical stability
        T max_val = input[0];

        for (std::size_t i = 1; i < input.size(); ++i) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }

        // Step 2: Calculate exponentials and their sum
        Tensor<T> output(input.shape());
        T sum_exp = T{};

        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum_exp += output[i];
        }

        // Step 3: Normalize to probabilities
        for (std::size_t i = 0; i < output.size(); ++i) {
            output[i] /= sum_exp;
        }

        return output;
    }

    Tensor<T> forward_two(const Tensor<T>& input) const {

        if (input.shape().size() != 2) {
            throw std::invalid_argument("the shape should be 2 dimentional");
        }

        std::size_t row = input.shape()[0];
        std::size_t col = input.shape()[1];

        if (row == 0 || col == 0) {
            throw std::invalid_argument("Empty tensor");
        }

       Tensor<T> output(input.shape());

        for (std::size_t i = 0; i < row; i++) {
            T max_val = input(i,0);


            for (std::size_t j = 0; j < col; j++) {
                if (input(i, j) > max_val) {
                    max_val = input(i, j);
                }
            }
            
            T sum_exp = T{};

            for (std::size_t j = 0; j < col; j++) {
                output(i, j) = std::exp(input(i, j) - max_val);
                sum_exp += output(i, j);
            }

            for (std::size_t j = 0; j< col; j++) {
                output(i, j) /= sum_exp;
            }
        }
        return output;
    }
};

} // namespace mini_infer
