#pragma once 

#include "mini_infer/tensor.hpp"


namespace mini_infer {

template<typename T>
class LayerNormalization {
    private:
    //
    public:
        Tensor<T> forward(const Tensor<T>& input) const;
        Tensor<T> forward_two(const Tensor<T>& input) const; 
};

}