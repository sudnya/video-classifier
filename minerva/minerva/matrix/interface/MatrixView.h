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
	const Dimension& size() const
	{
		return _size;
	}
	
	const Dimension& stride() const
	{
		return _stride;
	}

public:
	size_t linearAddress(const Dimension& d) const
	{
		return dotProduct(d, _stride);
	}

public:
	T& operator()(const Dimension& d) const
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
	: _data(static_cast<const T*>(matrix.data())), _size(matrix.size()), _stride(matrix.stride())
	{
	
	}

public:
	const Dimension& size() const
	{
		return _size;
	}
	
	const Dimension& stride() const
	{
		return _stride;
	}

public:
	size_t linearAddress(const Dimension& d) const
	{
		return dotProduct(d, _stride);
	}

public:
	const T& operator()(const Dimension& d) const
	{
		return _data[linearAddress(d)];
	}

private:
	const T*  _data;
	Dimension _size;
	Dimension _stride;

};


}
}




