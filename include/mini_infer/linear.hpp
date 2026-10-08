#pragma once 

#include "mini_infer/tensor.hpp"

namespace mini_infer {


    template<typename T>
    class Linear {
        public:
            Linear(std::size_t in_features, std::size_t out_features, const Tensor<T>& weights, const Tensor<T>& bias);


            Tensor<T> forward(const Tensor<T>& input) const;

        private:
            // parameters for the linear layer

            // Y = XW +
            Tensor<T> weights_;
            Tensor<T> bias_;
    };

}