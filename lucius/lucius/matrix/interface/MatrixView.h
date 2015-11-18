#pragma once

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Dimension.h>
#include <lucius/matrix/interface/DimensionTransformations.h>

#include <lucius/parallel/interface/cuda.h>

namespace lucius
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

    CUDA_DECORATOR MatrixView(T* data, const Dimension& size, const Dimension& stride)
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
CUDA_DECORATOR ConstMatrixView<T> slice(const ConstMatrixView<T>& input, const Dimension& begin, const Dimension& end)
{
    return ConstMatrixView<T>(&input(begin), end-begin, input.stride());
}

template<typename T>
CUDA_DECORATOR MatrixView<T> slice(const MatrixView<T>& input, const Dimension& begin, const Dimension& end)
{
    return MatrixView<T>(&input(begin), end-begin, input.stride());
}

}
}




