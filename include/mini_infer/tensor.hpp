#pragma once

#include <cstddef>
#include <vector>

namespace mini_infer {

// A dense, n-dimensional array of elements of type T.
//
// Scaffold only: this declares the intended storage layout so the project
// compiles. Constructors, accessors, and operations come in later phases.
template <typename T>
class Tensor {

public:
    // Tensor<T> t;
    Tensor() = default;

    Tensor(const std::vector<std::size_t>& shape);

    T& operator()(std::size_t row, std::size_t col);

    const T& operator()(std::size_t row, std::size_t col) const;

private:
    // tensor is row major right now 
    std::vector<T> data_;              // elements, flattened
    std::vector<std::size_t> shape_;   // size of each dimension


};

}  // namespace mini_infer
