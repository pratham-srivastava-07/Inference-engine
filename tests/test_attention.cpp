#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mini_infer/dot_attention.hpp"

using mini_infer::Tensor;
using mini_infer::ScaledDotProductAttention;

constexpr float EPS = 1e-4f;

bool near(float a, float b) {
    return std::abs(a - b) < EPS;
}

// Fill a 2D tensor row by row from a flat list of values.
Tensor<float> make2d(std::size_t rows, std::size_t cols, const std::vector<float>& values) {
    Tensor<float> t({rows, cols});

    for (std::size_t i = 0; i < values.size(); i++) {
        t[i] = values[i];
    }

    return t;
}

// Test 1: hand-computed values
//
// Q = K = identity, d_k = 2
// scores = Q * K^T / sqrt(2) = [[0.7071, 0], [0, 0.7071]]
// softmax row 0 = [0.669757, 0.330243], row 1 = [0.330243, 0.669757]
// output = weights * V
void test_attention_values() {
    Tensor<float> query = make2d(2, 2, {1, 0, 0, 1});
    Tensor<float> key   = make2d(2, 2, {1, 0, 0, 1});
    Tensor<float> value = make2d(2, 2, {1, 2, 3, 4});

    ScaledDotProductAttention<float> attention;
    Tensor<float> output = attention.forward(query, key, value);

    assert((output.shape() == std::vector<std::size_t>{2, 2}));

    assert(near(output(0, 0), 1.660486f));
    assert(near(output(0, 1), 2.660486f));
    assert(near(output(1, 0), 2.339514f));
    assert(near(output(1, 1), 3.339514f));

    std::cout << "Attention values test passed!\n";
}

// Test 2: output shape is [seq_q, d_v], even when nothing is square
void test_attention_shape() {
    Tensor<float> query({1, 2});   // seq_q = 1, d_k = 2
    Tensor<float> key({3, 2});     // seq_k = 3, d_k = 2
    Tensor<float> value({3, 4});   // seq_k = 3, d_v = 4

    ScaledDotProductAttention<float> attention;
    Tensor<float> output = attention.forward(query, key, value);

    assert((output.shape() == std::vector<std::size_t>{1, 4}));

    std::cout << "Attention shape test passed!\n";
}

// Test 3: if all scores are equal, every key gets the same weight,
// so each output row is the average of the value rows
void test_attention_uniform_weights() {
    Tensor<float> query({2, 2});   // all zeros -> every score is 0
    Tensor<float> key   = make2d(3, 2, {1, 2, 3, 4, 5, 6});
    Tensor<float> value = make2d(3, 2, {1, 2, 3, 4, 5, 6});

    ScaledDotProductAttention<float> attention;
    Tensor<float> output = attention.forward(query, key, value);

    // average of rows [1,2], [3,4], [5,6] = [3, 4]
    for (std::size_t i = 0; i < 2; i++) {
        assert(near(output(i, 0), 3.0f));
        assert(near(output(i, 1), 4.0f));
    }

    std::cout << "Attention uniform weights test passed!\n";
}

// Test 4: a query that strongly matches one key should copy that key's value row
void test_attention_strong_match() {
    Tensor<float> query = make2d(1, 2, {100, 0});
    Tensor<float> key   = make2d(2, 2, {1, 0, 0, 1});   // key 0 matches the query
    Tensor<float> value = make2d(2, 2, {7, 8, -7, -8});

    ScaledDotProductAttention<float> attention;
    Tensor<float> output = attention.forward(query, key, value);

    assert(near(output(0, 0), 7.0f));
    assert(near(output(0, 1), 8.0f));

    std::cout << "Attention strong match test passed!\n";
}

// Test 5: inputs must not be modified
void test_attention_does_not_modify_inputs() {
    Tensor<float> query = make2d(2, 2, {1, 2, 3, 4});
    Tensor<float> key   = make2d(2, 2, {5, 6, 7, 8});
    Tensor<float> value = make2d(2, 2, {9, 10, 11, 12});

    ScaledDotProductAttention<float> attention;
    (void)attention.forward(query, key, value);

    for (std::size_t i = 0; i < 4; i++) {
        assert(query[i] == static_cast<float>(i + 1));
        assert(key[i]   == static_cast<float>(i + 5));
        assert(value[i] == static_cast<float>(i + 9));
    }

    std::cout << "Attention input unchanged test passed!\n";
}

// Returns true if forward() throws std::invalid_argument for these inputs.
bool throws_invalid_argument(const Tensor<float>& query, const Tensor<float>& key,
                             const Tensor<float>& value) {
    ScaledDotProductAttention<float> attention;

    try {
        (void)attention.forward(query, key, value);
    } catch (const std::invalid_argument&) {
        return true;
    }

    return false;
}

// Test 6: invalid shapes are rejected
void test_attention_invalid_shapes() {
    // query and key have different d_k (2 vs 3)
    assert(throws_invalid_argument(Tensor<float>({2, 2}), Tensor<float>({2, 3}), Tensor<float>({2, 2})));

    // key and value have different sequence lengths (2 vs 3)
    assert(throws_invalid_argument(Tensor<float>({2, 2}), Tensor<float>({2, 2}), Tensor<float>({3, 2})));

    // 1D query
    assert(throws_invalid_argument(Tensor<float>({2}), Tensor<float>({2, 2}), Tensor<float>({2, 2})));

    // empty tensors
    assert(throws_invalid_argument(Tensor<float>({0, 2}), Tensor<float>({0, 2}), Tensor<float>({0, 2})));

    std::cout << "Attention invalid shapes test passed!\n";
}

int main() {
    test_attention_values();
    test_attention_shape();
    test_attention_uniform_weights();
    test_attention_strong_match();
    test_attention_does_not_modify_inputs();
    test_attention_invalid_shapes();

    std::cout << "All attention tests passed!\n";
    return 0;
}
