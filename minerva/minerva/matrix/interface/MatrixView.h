#pragma once

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/Dimension.h>
#include <minerva/matrix/interface/MatrixTransformations.h>

namespace minerva
{
namespace matrix
{

template<typename T>
class MatrixView
{
public:
    MatrixView(Matrix& matrix)
    : _data(static_cast<T*>(matrix.data())), _size(matrix.size()), _stride(matrix.stride())
    {

    }

public:
    CUDA_DECORATOR const Dimension& size() const
    {
        return _size;
    }

    CUDA_DECORATOR const Dimension& stride() const
    {
        return _stride;
    }

public:
    CUDA_DECORATOR size_t linearAddress(const Dimension& d) const
    {
        return dotProduct(d, _stride);
    }

public:
    CUDA_DECORATOR T& operator()(const Dimension& d) const
    {
        return _data[linearAddress(d)];
    }

private:
    T*        _data;
    Dimension _size;
    Dimension _stride;

};

template<typename T>
class ConstMatrixView
{
public:
    ConstMatrixView(const Matrix& matrix)
    : ConstMatrixView(static_cast<const T*>(matrix.data()), matrix.size(), matrix.stride())
    {

    }

    CUDA_DECORATOR ConstMatrixView(const T* data, const Dimension& size, const Dimension& stride)
    : _data(data), _size(size), _stride(stride)
    {

    }

public:
    CUDA_DECORATOR const Dimension& size() const
    {
        return _size;
    }

    CUDA_DECORATOR const Dimension& stride() const
    {
        return _stride;
    }

public:
    CUDA_DECORATOR size_t elements() const
    {
        return _size.product();
    }

public:
    CUDA_DECORATOR size_t linearAddress(const Dimension& d) const
    {
        return dotProduct(d, _stride);
    }

public:
    CUDA_DECORATOR const T& operator()(const Dimension& d) const
    {
        return _data[linearAddress(d)];
    }

private:
    const T*  _data;
    Dimension _size;
    Dimension _stride;

};

template<typename T>
ConstMatrixView<T> slice(const ConstMatrixView<T>& input, const Dimension& begin, const Dimension& end)
{
    return ConstMatrixView<T>(&input(begin), end-begin, input.stride());
}

}
}




