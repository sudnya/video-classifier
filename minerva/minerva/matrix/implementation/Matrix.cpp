/*	\file   Matrix.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Matrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/Allocation.h>
#include <minerva/matrix/interface/MatrixTransformations.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cmath>

namespace minerva
{

namespace matrix
{

static Dimension linearStride(const Dimension& size)
{
	Dimension stride;
	
	size_t step = 1;
	
	for (auto sizeStep : size)
	{
		stride.push_back(step);
		step *= sizeStep;
	}
	
	return stride;
}
	
Matrix::Matrix()
: _data_begin(nullptr), _precision(SinglePrecision())
{

}

Matrix::Matrix(const Dimension& size)
: _allocation(std::make_shared<Allocation>(size.product() * SinglePrecision().size())),
  _data_begin(_allocation->data()),
  _size(size), _stride(linearStride(size)), _precision(SinglePrecision())
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

std::shared_ptr<Allocation> Matrix::allocation()
{
	return _allocation;
}

std::string Matrix::toString() const
{
	auto matrix = reshape(*this, {size()[0], size()[1]});

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

static size_t getOffset(const Dimension& stride, const Dimension& position)
{
	size_t offset = 0;
	
	for(auto i = 0; i < stride.size(); ++i)
	{
		offset += stride[i] * position[i];
	}
	
	return offset;
}

static void* getAddress(const Dimension& stride, const Dimension& position, void* data, const Precision& precision)
{
	size_t offset = getOffset(stride, position);
	
	uint8_t* address = static_cast<uint8_t*>(data);
	
	return address + precision.size() * offset;
}
	
FloatReference Matrix::operator[](const Dimension& d)
{
	return FloatReference(getAddress(stride(), d, _data_begin, precision), _precision);
}

ConstFloatReference Matrix::operator[](const Dimension& d) const
{
	return ConstFloatReference(getAddress(stride(), d, _data_begin, precision), _precision);
}

}

}


