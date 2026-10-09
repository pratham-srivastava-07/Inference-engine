#pragma once 

#include "mini_infer/tensor.hpp"


namespace mini_infer {


    template<typename T>
    class Relu {
        // activation functions introduce non linearity and 
        // activation function used as f(x) = max(0, x);

        public:
            Tensor<T> forward(const Tensor<T>& input) const;
    };

}