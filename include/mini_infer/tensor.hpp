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

    void computeStrides();

    T& operator()(const std::vector<std::size_t>& coordinates);

    const T& operator()(const std::vector<std::size_t>& coordinates) const;

    Tensor<T> operator+(const Tensor<T>& vec) const;

    Tensor<T> operator*(const Tensor<T>& other) const;

    const std::vector<std::size_t>& shape() const;

    // a public size method and operator[] overloading for tensor 

    const T& operator[](std::size_t index);

    T& operator[](std::size_t index);

    std::size_t size() noexcept;

    

private:
    // tensor is row major right now 
    std::vector<T> data_;              // elements, flattened
    std::vector<std::size_t> shape_;   // size of each dimension

    std::vector<std::size_t> strides;


};

}  // namespace mini_infer
