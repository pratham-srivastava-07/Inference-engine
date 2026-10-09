#pragma once 

#include "mini_infer/tensor.hpp"


namespace mini_infer {


    template<typename T>
    class Relu {

        public:
            Tensor<T> forward(const Tensor<T> input) const;
    };

}