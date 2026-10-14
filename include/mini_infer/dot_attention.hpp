#pragma once

#include "mini_infer/tensor.hpp"

namespace mini_infer {

template<typename T>
class ScaledDotProductAttention {
    private:
    public:
        Tensor<T> forward(const Tensor<T>& query, const Tensor<T>& key, const Tensor<T>& value);
};    
}