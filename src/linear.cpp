// #include <bits/stdc++.h>
#include<stdexcept>
#include<vector>

#include "mini_infer/linear.hpp"

using namespace std;

namespace mini_infer {

    template<typename T>
    Tensor<T> Linear<T>::forward(const Tensor<T>& input) const {
        Tensor<T> output;

        output = input * (this->weights_);

        // add bias to each row 
        for (std::size_t i=0; i<output.shape()[0]; i++) {
            for (std::size_t j=0; j<output.shape()[1]; j++) {
                output(i, j) += bias_({j});
            }
        }

        return output;
    }
    template<typename T>
    Linear<T>::Linear(std::size_t in_features, std::size_t out_features, 
    const Tensor<T>& weights, const Tensor<T>& bias) : weights_(weights), bias_(bias) {


        if (weights_.shape() != std::vector<std::size_t>{in_features, out_features}) {
            throw invalid_argument("Invalid structure of linear weights");
        }

        if (bias_.shape() != std::vector<std::size_t>{out_features}) {
            throw invalid_argument("Invalid bias shape");
        }
    }

    template class Linear<float>;

}