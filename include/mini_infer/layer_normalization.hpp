#pragma once 

#include "mini_infer/tensor.hpp"


namespace mini_infer {

template<typename T>
class LayerNormalization {
    private:
    Tensor<T> gamma_;
    Tensor<T> beta_;
    //
    public:
        LayerNormalization() = default;
        LayerNormalization(const Tensor<T>& gamma, const Tensor<T>& beta);
        Tensor<T> forward(const Tensor<T>& input) const;
        Tensor<T> forward_two(const Tensor<T>& input) const; 
};

}