/*	\file   Matrix.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Matrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixImplementation.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cmath>

namespace minerva
{

namespace matrix
{
	
Matrix::Matrix()
: _precision(SinglePrecision())
{

}

Matrix::Matrix(const Dimension& size)
: _size(size), _stride(linearStride(size)), _precision(SinglePrecision()), _allocation(std::make_shared<Allocation>(size.product() * SinglePrecision().size()))
{

}

Matrix::Matrix(const Dimension& size, const Dimension& stride)
: _size(size), _stride(stride), _precision(SinglePrecision()), _allocation(std::make_shared<Allocation>(size.product() * SinglePrecision().size()))
{

}

Matrix::Matrix(const Dimension& size, const Dimension& stride, const Precision& precision)
: _size(size), _stride(stride), _precision(precision), _allocation(std::make_shared<Allocation>(size.product() * precision.size()))
{

}

Matrix::Matrix(const Dimension& size, const Dimension& stride, const Precision& precision, const std::shared_ptr<Allocation>& allocation)
: _size(size), _stride(stride), _precision(precision), _allocation(std::make_shared<Allocation>(size.product() * precision.size()))
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
	return flatten(size());
}

std::shared_ptr<Allocation> Matrix::allocation()
{
	return _allocation;
}

std::string Matrix::toString() const
{
	auto matrix = reshape(*this, {matrix.size()[0], matrix.size()[1]});

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
	
Matrix::FloatReference Matrix::operator[](const Dimension& d)
{
	return FloatReference(flatten(d));
}

Matrix::ConstFloatReference Matrix::operator[](const Dimension& d) const
{
	return ConstFloatReference(flatten(d));
}

}

}


