#include <cmath>
#include <stdexcept>

#include "mini_infer/dot_attention.hpp"


#include "softmax.cpp"

namespace mini_infer {

// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//
//   query: [seq_q, d_k]
//   key:   [seq_k, d_k]
//   value: [seq_k, d_v]
//   result: [seq_q, d_v]
template<typename T>
Tensor<T> ScaledDotProductAttention<T>::forward(const Tensor<T>& query,
    const Tensor<T>& key, const Tensor<T>& value)
{
    if (query.shape().size() != 2 || key.shape().size() != 2 || value.shape().size() != 2) {
        throw std::invalid_argument("Attention requires 2D query, key and value");
    }

    if (query.size() == 0 || key.size() == 0 || value.size() == 0) {
        throw std::invalid_argument("Attention requires non-empty tensors");
    }

    if (query.shape()[1] != key.shape()[1]) {
        throw std::invalid_argument("Query and key must have the same feature size (d_k)");
    }

    if (key.shape()[0] != value.shape()[0]) {
        throw std::invalid_argument("Key and value must have the same sequence length");
    }

    // Step 1: dot product of every query row with every key row -> [seq_q, seq_k]
    Tensor<T> scores = query * key.transpose();

    // Step 2: scale by 1/sqrt(d_k) so large d_k doesn't push softmax into
    // regions where one weight is ~1 and the rest are ~0
    const T scale = static_cast<T>(1) / std::sqrt(static_cast<T>(query.shape()[1]));

    for (std::size_t i = 0; i < scores.size(); i++) {
        scores[i] *= scale;
    }

    // Step 3: each row becomes a probability distribution over the keys
    Softmax<T> softmax;
    Tensor<T> weights = softmax.forward_two(scores);

    // Step 4: weighted sum of the value rows -> [seq_q, d_v]
    return weights * value;
}

template class ScaledDotProductAttention<float>;

}  // namespace mini_infer
