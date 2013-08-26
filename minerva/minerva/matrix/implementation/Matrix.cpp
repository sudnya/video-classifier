/*	\file   Matrix.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Matrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cmath>

namespace minerva
{

namespace matrix
{

Matrix::Matrix(size_t r, size_t c, const FloatVector& d)
: _rows(r), _columns(c), _data(d)
{
    resize(rows(), columns());
}

Matrix::iterator Matrix::begin()
{
	return _data.begin();
}

Matrix::const_iterator Matrix::begin() const
{
	return _data.begin();
}

Matrix::iterator Matrix::end()
{
	return _data.end();
}

Matrix::const_iterator Matrix::end() const
{
	return _data.end();
}

float& Matrix::operator[](size_t index)
{
	return _data[index];
}

const float& Matrix::operator[](size_t index) const
{
	return _data[index];
}

float& Matrix::operator()(size_t row, size_t column)
{
	size_t position = getPosition(row, column);

	return _data[position];
}

const float& Matrix::operator()(size_t row, size_t column) const
{
	size_t position = getPosition(row, column);

	return _data[position];
}

size_t Matrix::size() const
{
	return _data.size();
}

bool Matrix::empty() const
{
    return _data.empty();
}

void Matrix::resize(size_t rows, size_t columns)
{
	_data.resize(rows * columns);
	
	_rows	 = rows;
	_columns = columns;
}

Vector Matrix::getColumn(size_t number) const
{
	assert(number < columns());

	Vector column(rows());
	
	// TODO faster
	for(size_t row = 0; row != rows(); ++row)
	{
		column[row] = (*this)(row, number);
	}
	
	return column;
}

Vector Matrix::getRow(size_t number) const
{
	assert(number < rows());

	Vector row(columns());
	
	size_t position = getPosition(number, 0);
	
	std::memcpy(row.data(), &_data[position], sizeof(float) * columns());
	
	return row;
}

size_t Matrix::columns() const
{
	return _columns;
}

size_t Matrix::rows() const
{
	return _rows;
}

size_t Matrix::getPosition(size_t row, size_t column) const
{
	return row * columns() + column;
}

Matrix Matrix::multiply(float f) const
{
	Matrix result(*this);
	
	// TODO: faster
	for(auto& value : result._data)
	{
		value += f;
	}
	
	return result;
}

Matrix Matrix::multiply(const Matrix& m) const
{
	assert(columns() == m.rows());

	Matrix result(rows(), m.columns());
	/*
	const size_t blockSize = 32;
	
	size_t blocksX = rows()    / blockSize;
	size_t blocksY = columns() / blockSize;
	
	for(size_t blockX = 0; blockX != blocksX; ++blockX)
	{
		for(size_t blockY = 0; blockY != blocksY; ++blockY)
		{
			
		}
	}*/
	
	
	
	// TODO: much faster
	for(size_t row = 0; row != result.rows(); ++row)
	{
		for(size_t column = 0; column != result.columns(); ++column)
		{
			float value = 0.0f;
			
			for(size_t inner = 0; inner != columns(); ++inner)
			{
				value += (*this)(row, inner) * m(inner, column);
			}

            result(row, column) = value;
		}
	}
	
	return result;
}

Matrix Matrix::elementMultiply(const Matrix& m) const
{
	assert(m.rows()    == rows()   );
	assert(m.columns() == columns());

    Matrix result(*this);

	// TODO: faster
	auto rValue = result._data.begin();
	for(auto value = m.begin(); value != m.end(); ++value, ++rValue)
	{
		*rValue *= *value;
	}

    return result;
}

Matrix Matrix::add(float f) const
{
	Matrix result(*this);
	
	// TODO: faster
	for(auto& value : result._data)
	{
		value += f;
	}
	
	return result;
}

Matrix Matrix::add(const Matrix& m) const
{
	assert(m.rows()    == rows());
	assert(m.columns() == columns());

	Matrix result(*this);
	
	// TODO: faster
	auto rValue = result._data.begin();
	for(auto value = m.begin(); value != m.end(); ++value, ++rValue)
	{
		*rValue += *value;
	}
	
	return result;
}

Matrix Matrix::subtract(const Matrix& m) const
{
	assert(m.rows()    == rows());
	assert(m.columns() == columns());

	Matrix result(*this);
	
	// TODO: faster
	auto rValue = result._data.begin();
	for(auto value = m.begin(); value != m.end(); ++value, ++rValue)
	{
		*rValue -= *value;
	}
	
	return result;
}

Matrix Matrix::slice(size_t startRow, size_t startColumn,
	size_t rows, size_t columns) const
{
	Matrix result(rows, columns);
	
	assert(startRow    + rows    <= this->rows()   );
	assert(startColumn + columns <= this->columns());
	
	// TODO: faster
	for(size_t row = 0; row != rows; ++row)
	{
		for(size_t column = 0; column != columns; ++column)
		{
			result(row, column) = (*this)(row + startRow, column + startColumn);
		}
	}
	
	return result;
}

Matrix Matrix::transpose() const
{
	Matrix result(columns(), rows());
	
	// TODO: faster
	for(size_t row = 0; row != rows(); ++row)
	{
		for(size_t column = 0; column != columns(); ++column)
		{
			result(column, row) = (*this)(row, column);
		}
	}
	
	return result;
}

Matrix Matrix::append(const Matrix& m) const
{
	assert(empty() || (rows() == m.rows()));

    size_t resultRows = rows();

    if(empty())
    {
        resultRows = m.rows();
    }

	Matrix result(resultRows, columns() + m.columns());
	
	// Copy rows from the original and appended matrices
	for(size_t row = 0; row != resultRows; ++row)
	{
		size_t originalPosition = getPosition(row, 0);
		size_t newPosition = result.getPosition(row, 0);
	
		std::memcpy(&result._data[newPosition], &_data[originalPosition],
			columns() * sizeof(float));

		size_t appendedPosition = m.getPosition(row, 0);
		newPosition += columns();
		
		std::memcpy(&result._data[newPosition], &m._data[appendedPosition],
			m.columns() * sizeof(float));
	}
	
	return result;
}

Matrix Matrix::log() const
{
    Matrix result(*this);
	
	result.logSelf();

    return result;
}

Matrix Matrix::negate() const
{
    Matrix result(*this);
	
	result.negateSelf();

    return result;
}

Matrix Matrix::sigmoid() const
{
    Matrix result(*this);
	
	result.sigmoidSelf();

    return result;
}

void Matrix::appendRowData(const FloatVector& f)
{
	assert(f.size() == columns());

	_data.insert(_data.end(), f.begin(), f.end());

    ++_rows;
}

void Matrix::setRowData(size_t row, const FloatVector& f)
{
    assert(row < rows());
    assert(f.size() == columns());

    size_t position = getPosition(row, 0);

    std::memcpy(&_data[position], f.data(), sizeof(float) * columns());
}

void Matrix::negateSelf()
{
	for(auto& f : _data)
	{
		f = -f;
	}
}

void Matrix::logSelf()
{
	for(auto& f : _data)
	{
		f = std::log(f);
	}
}

static float sigmoid(float v)
{
    return 1.0f / (1 + std::exp(-v)); 
}

void Matrix::sigmoidSelf()
{
	for(auto& f : _data)
	{
		f = matrix::sigmoid(f);
	}
}

void Matrix::transposeSelf()
{
	*this = transpose();
}

void* Matrix::data()
{
	return _data.data();
}

const void* Matrix::data() const
{
	return _data.data();
}

std::string Matrix::toString(size_t maxRows, size_t maxColumns) const
{
    std::stringstream stream;

    stream << "[ ";

    for(size_t row = 0; row != std::min(rows(), maxRows); ++row)
    {
        for(size_t column = 0; column != std::min(columns(), maxColumns); ++column)
        {
            stream << (*this)(row, column) << " ";
        }
        
        stream << "\n ";
    }

    stream << "]\n";

    return stream.str();
}

}

}


