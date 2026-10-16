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

    template<typename T>
    Tensor<T> Relu<T>::forward_two_dimentional(const Tensor<T>& input) const 
    {
        if (input.shape().size() != 2 || input.size() == 0) 
        {
            throw std::invalid_argument(
                "Invalid dimentions for 2d relu"
            );
        }

        Tensor<T> output(input.shape());

        std::size_t rows = input.shape()[0];
        std::size_t cols = input.shape()[1];

        for (std::size_t i = 0; i < rows; i++) 
        {
            for (std::size_t j = 0; j < cols; j++)
            {
                if (input(i, j) < T{})
                {
                    output(i, j) = T{};
                }

                else {
                    output(i, j) = input(i, j);
                }
            }
        }
        return output;
    }
}
