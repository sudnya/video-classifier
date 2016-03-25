/*    \file   MatrixVector.cpp
    \date   Sunday August 11, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the MatrixVector class.
*/

// Lucius Includes
#include <lucius/matrix/interface/MatrixVector.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace matrix
{

MatrixVector::MatrixVector()
{

}

MatrixVector::MatrixVector(const DimensionVector& sizes)
: MatrixVector(sizes, SinglePrecision())
{

}

MatrixVector::MatrixVector(const DimensionVector& sizes, const Precision& precision)
{
    for(auto& size : sizes)
    {
        push_back(Matrix(size, precision));
    }
}

MatrixVector::MatrixVector(std::initializer_list<Matrix> l)
: _matrix(l)
{

}

MatrixVector::MatrixVector(const MatrixVector& m) = default;

MatrixVector::MatrixVector(MatrixVector&& m) = default;

MatrixVector& MatrixVector::operator=(const MatrixVector&  ) = default;
MatrixVector& MatrixVector::operator=(MatrixVector&& ) = default;

MatrixVector::reference_type MatrixVector::operator[](size_t i)
{
    return _matrix[i];
}

MatrixVector::const_reference_type MatrixVector::operator[](size_t i) const
{
    return _matrix[i];
}

bool MatrixVector::empty() const
{
    return _matrix.empty();
}

size_t MatrixVector::size() const
{
    return _matrix.size();
}

MatrixVector::DimensionVector MatrixVector::sizes() const
{
    DimensionVector sizes;

    for(auto& matrix : *this)
    {
        sizes.push_back(matrix.size());
    }

    return sizes;
}

void MatrixVector::reserve(size_t size)
{
    _matrix.reserve(size);
}

void MatrixVector::resize(size_t size)
{
    _matrix.resize(size);
}

void MatrixVector::clear()
{
    _matrix.clear();
}

MatrixVector::iterator MatrixVector::begin()
{
    return _matrix.begin();
}

MatrixVector::const_iterator MatrixVector::begin() const
{
    return _matrix.begin();
}

MatrixVector::iterator MatrixVector::end()
{
    return _matrix.end();
}

MatrixVector::const_iterator MatrixVector::end() const
{
    return _matrix.end();
}

MatrixVector::reverse_iterator MatrixVector::rbegin()
{
    return _matrix.rbegin();
}

MatrixVector::const_reverse_iterator MatrixVector::rbegin() const
{
    return _matrix.rbegin();
}

MatrixVector::reverse_iterator MatrixVector::rend()
{
    return _matrix.rend();
}

MatrixVector::const_reverse_iterator MatrixVector::rend() const
{
    return _matrix.rend();
}

void MatrixVector::push_back(const Matrix& m)
{
    _matrix.push_back(m);
}

void MatrixVector::push_back(Matrix&& m)
{
    _matrix.push_back(std::move(m));
}

void MatrixVector::push_front(const Matrix&  m)
{
    Vector temp;

    temp.push_back(m);

    for(auto& m : _matrix)
    {
        temp.push_back(m);
    }

    _matrix = std::move(temp);
}

void MatrixVector::push_front(Matrix&& m)
{
    Vector temp;

    temp.push_back(std::move(m));

    for(auto& m : _matrix)
    {
        temp.push_back(m);
    }

    _matrix = std::move(temp);
}

void MatrixVector::push_back(MatrixVector&& v)
{
    for(auto& m : v)
    {
        _matrix.push_back(std::move(m));
    }
}

void MatrixVector::push_back(const MatrixVector& v)
{
    for(auto& m : v)
    {
        _matrix.push_back(m);
    }
}

void MatrixVector::pop_back()
{
    _matrix.pop_back();
}

void MatrixVector::pop_front()
{
    assert(!empty());

    Vector newMatrix;

    for(auto i = ++begin(); i != end(); ++i)
    {
        newMatrix.push_back(std::move(*i));
    }

    _matrix = std::move(newMatrix);
}

Matrix& MatrixVector::back()
{
    return _matrix.back();
}

const Matrix& MatrixVector::back() const
{
    return _matrix.back();
}

Matrix& MatrixVector::front()
{
    return _matrix.front();
}

const Matrix& MatrixVector::front() const
{
    return _matrix.front();
}

std::string MatrixVector::toString() const
{
    if(empty())
    {
        return "[]";
    }

    return front().toString();
}

bool MatrixVector::operator==(const MatrixVector& l) const
{
    return _matrix == l._matrix;
}

bool MatrixVector::operator!=(const MatrixVector& l) const
{
    return _matrix != l._matrix;
}

}

}


