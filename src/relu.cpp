#include "mini_infer/relu.hpp"

namespace mini_infer {
    template<typename T>
    Tensor<T> Relu<T>::forward(const Tensor<T>& input) const {
        Tensor<T> output(input.shape());

        for (std::size_t i=0; i<input.size(); i++) {
            if (input[i] < T{}) {
                output[i] = T{}; // type generic way of representing 0
            }


            else {
                output[i] = input[i];
            }
        }
        return output;
    }

    // Explicit instantiation (see tensor.cpp)
    template class Relu<float>;
}
