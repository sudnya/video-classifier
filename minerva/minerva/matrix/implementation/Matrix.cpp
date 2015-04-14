/*    \file   Matrix.cpp
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the Matrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/Allocation.h>
#include <minerva/matrix/interface/MatrixTransformations.h>
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cmath>

namespace minerva
{

namespace matrix
{

Matrix::Matrix()
: _data_begin(nullptr), _precision(SinglePrecision())
{

}

Matrix::Matrix(std::initializer_list<size_t> i)
: Matrix(Dimension(i))
{

}

Matrix::Matrix(const Dimension& size)
: _allocation(std::make_shared<Allocation>(size.product() * SinglePrecision().size())),
  _data_begin(_allocation->data()),
  _size(size), _stride(linearStride(size)), _precision(SinglePrecision())
{

}

Matrix::Matrix(const Dimension& size, const Precision& precision)
: _allocation(std::make_shared<Allocation>(size.product() * precision.size())),
  _data_begin(_allocation->data()),
  _size(size), _stride(linearStride(size)), _precision(precision)
{

}

Matrix::Matrix(const Dimension& size, const Dimension& stride)
: _allocation(std::make_shared<Allocation>(size.product() * SinglePrecision().size())),
  _data_begin(_allocation->data()),
  _size(size), _stride(stride), _precision(SinglePrecision()) 
{

}

Matrix::Matrix(const Dimension& size, const Dimension& stride, const Precision& precision)
: _allocation(std::make_shared<Allocation>(size.product() * precision.size())),
  _data_begin(_allocation->data()),
  _size(size), _stride(stride), _precision(precision)
{

}

Matrix::Matrix(const Dimension& size, const Dimension& stride, const Precision& precision, const std::shared_ptr<Allocation>& allocation)
: _allocation(std::make_shared<Allocation>(size.product() * precision.size())),
  _data_begin(_allocation->data()),
  _size(size), _stride(stride), _precision(precision)
{

}

Matrix::Matrix(const Dimension& size, const Dimension& stride, const Precision& precision,
               const std::shared_ptr<Allocation>& allocation, void* start)
: _allocation(std::make_shared<Allocation>(size.product() * precision.size())),
  _data_begin(start),
  _size(size), _stride(stride), _precision(precision)
{

}

Matrix::~Matrix()
{

}

const Dimension& Matrix::size() const
{
    return _size;
}

const Dimension& Matrix::stride() const
{
    return _stride;
}

const Precision& Matrix::precision() const
{
    return _precision;
}

size_t Matrix::elements() const
{
    return size().product();
}
    
FloatIterator Matrix::begin()
{
    return FloatIterator(precision(), size(), stride(), zeros(size()), _data_begin);
}

FloatIterator Matrix::end()
{
    return FloatIterator(precision(), size(), stride(), size(), _data_begin);
}

ConstFloatIterator Matrix::begin() const
{
    return ConstFloatIterator(precision(), size(), stride(), zeros(size()), _data_begin);
}

ConstFloatIterator Matrix::end() const
{
    return FloatIterator(precision(), size(), stride(), zeros(size()), _data_begin);
}

std::shared_ptr<Allocation> Matrix::allocation()
{
    return _allocation;
}

void* Matrix::data()
{
    return _data_begin;
}

const void* Matrix::data() const
{
    return _data_begin;
}
    
bool Matrix::isContiguous() const
{
    return linearStride(size()) == stride();
}

bool Matrix::isLeadingDimensionContiguous() const
{
    return stride()[0] == 1;
}

std::string Matrix::toString() const
{
    size_t rows = 1;
    
    if(size().size() > 0)
    {
        rows = size()[0];
    }
    
    size_t columns = 1;
    
    if(size().size() > 1)
    {
        columns = size()[1];
    }

    auto matrix = reshape(*this, {rows, columns});

    std::stringstream stream;

    stream << shapeString() << " ";

    stream << "[ ";
    
    size_t maxRows    = 10;
    size_t maxColumns = 10;

    size_t finalRow = std::min(matrix.size()[0], maxRows);

    for(size_t row = 0; row != finalRow; ++row)
    {
        size_t finalColumn = std::min(matrix.size()[1], maxColumns);

        for(size_t column = 0; column != finalColumn; ++column)
        {
            stream << matrix(row, column) << " ";
        }
        
        if(row + 1 != finalRow) stream << "\n ";
    }

    stream << "]\n";

    return stream.str();
}

std::string Matrix::debugString() const
{
    return toString();
}

std::string Matrix::shapeString() const
{
    return size().toString();
}
    
FloatReference Matrix::operator[](const Dimension& d)
{
    return FloatReference(precision(), getAddress(stride(), d, _data_begin, precision()));
}

ConstFloatReference Matrix::operator[](const Dimension& d) const
{
    return ConstFloatReference(precision(), getAddress(stride(), d, _data_begin, precision()));
}
    
bool Matrix::operator==(const Matrix& m) const
{
    if(size() != m.size())
    {
        return false;
    }
    
    if(precision() != m.precision())
    {
        return false;
    }

    return reduce(apply(*this, m, NotEqual()), {}, Add())[0] == 0;
}

bool Matrix::operator!=(const Matrix& m) const
{
    return !(*this == m);
}

}

}


