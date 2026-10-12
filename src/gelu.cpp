#include "mini_infer/gelu.hpp"

#include <cmath>
#include <stdexcept>

namespace mini_infer {
 
    
template<typename T>
Tensor<T> Gelu<T>::forward(const Tensor<T>& input) const {

    // formulae to be impl : GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    if (input.size() == 0)
    {
        throw std::invalid_argument("GELU requires a non-empty tensor");
    }

    Tensor<T> output(input.shape());

    const T sqrt_two_over_pi = static_cast<T>(0.7978845608028654);
    const T coeff = static_cast<T>(0.044715);
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1);

    for (std::size_t i = 0; i < input.size(); ++i) {
        const T x = input[i];

        const T x_cubed = x * x * x;
        const T inner = x + coeff * x_cubed;
        const T middle = sqrt_two_over_pi * inner;

        output[i] = half * x * (one + std::tanh(middle));
    }

    return output;

}
template class Gelu<float>;
}