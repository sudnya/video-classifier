/*	\file   AtlasMatrix.cpp
	\date   Monday September 2, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the AtlasMatrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/AtlasMatrix.h>

#include <minerva/matrix/interface/AtlasLibrary.h>

// Standard Library Includes
#include <cassert>
#include <cmath>
#include <random>
#include <ctime>

namespace minerva
{

namespace matrix
{

typedef MatrixImplementation Value;

AtlasMatrix::AtlasMatrix(size_t r, size_t c, const FloatVector& data)
: MatrixImplementation(r, c, data)
{
	resize(rows(), columns());
}

void AtlasMatrix::resize(size_t rows, size_t columns)
{
	_data.resize(rows * columns);
	
	_rows	 = rows;
	_columns = columns;
}

Value* AtlasMatrix::appendColumns(const Value* matrix) const
{
	auto m = dynamic_cast<const AtlasMatrix*>(matrix);	
	assert(m != nullptr);

	assert(empty() || (rows() == m->rows()));

    size_t resultRows = rows();

    if(empty())
    {
        resultRows = m->rows();
    }

	AtlasMatrix* result = new AtlasMatrix(resultRows,
		columns() + m->columns());
	
	// Copy rows from the original and appended matrices
	for(size_t row = 0; row != resultRows; ++row)
	{
		size_t originalPosition = getPosition(row, 0);
		size_t newPosition = result->getPosition(row, 0);
	
		std::memcpy(&result->_data[newPosition], &_data[originalPosition],
			columns() * sizeof(float));

		size_t appendedPosition = m->getPosition(row, 0);
		newPosition += columns();
		
		std::memcpy(&(*result)._data[newPosition], &m->_data[appendedPosition],
			m->columns() * sizeof(float));
	}
	
	return result;
}

Value* AtlasMatrix::appendRows(const Value* matrix) const
{
	auto m = dynamic_cast<const AtlasMatrix*>(matrix);	
	assert(m != nullptr);

	assert(empty() || (columns() == m->columns()));

    size_t resultColumns = columns();

    if(empty())
    {
        resultColumns = m->columns();
    }

	AtlasMatrix* result = new AtlasMatrix(rows() + m->rows(), resultColumns);
	
	// Copy rows from the original and appended matrices
	std::memcpy(&result->_data[0], &_data[0], size() * sizeof(float));
	std::memcpy(&result->_data[size()], &m->_data[0],
		m->size() * sizeof(float));
	
	return result;
}

Value* AtlasMatrix::transpose() const
{
	AtlasMatrix* result = new AtlasMatrix(columns(), rows());
	
	// Cache blocked to 16x16x4 = 1KB
	const size_t blockSize = 16;
		
	for(size_t row = 0; row < rows(); row += blockSize)
	{
		for(size_t column = 0; column < columns(); column += blockSize)
		{
			size_t rowLimit    = std::min(rows(),    row + blockSize   );
			size_t columnLimit = std::min(columns(), column + blockSize);
			
			for(size_t blockRow = row; blockRow < rowLimit; ++blockRow)
			{
				for(size_t blockColumn = column;
					blockColumn < columnLimit; ++blockColumn)
				{
					result->_data[result->getPosition(blockColumn, blockRow)] =
						_data[getPosition(blockRow, blockColumn)];
				}
			}
		}
	}
	
	return result;
}
 
Value* AtlasMatrix::multiply(const Value* matrix) const
{
	auto m = dynamic_cast<const AtlasMatrix*>(matrix);	
	assert(m != nullptr);
	assert(columns() == m->rows());

	AtlasMatrix* result = new AtlasMatrix(rows(), m->columns());
		
	AtlasLibrary::sgemm(AtlasLibrary::CblasRowMajor, AtlasLibrary::CblasNoTrans,
		AtlasLibrary::CblasNoTrans, result->rows(), result->columns(), columns(),
		1.0f, &_data[0], columns(), &m->_data[0], m->columns(), 0.0f,
		&result->_data[0], result->columns());
	
	return result;
}

Value* AtlasMatrix::multiply(float f) const
{
	AtlasMatrix* result = new AtlasMatrix(*this);
	
	// TODO: faster
	for(auto& value : result->_data)
	{
		value *= f;
	}
	
	return result;
}

Value* AtlasMatrix::elementMultiply(const Value* matrix) const
{
	auto m = dynamic_cast<const AtlasMatrix*>(matrix);	
	assert(m != nullptr);

	assert(m->rows()    == rows()   );
	assert(m->columns() == columns());

    AtlasMatrix* result = new AtlasMatrix(*this);

	// TODO: faster
	auto rValue = result->_data.begin();
	for(auto value = m->_data.begin(); value != m->_data.end();
		++value, ++rValue)
	{
		*rValue *= *value;
	}

    return result;
}

Value* AtlasMatrix::add(const Value* matrix) const
{
	auto m = dynamic_cast<const AtlasMatrix*>(matrix);	
	assert(m != nullptr);
	
	assert(m->rows()    == rows());
	assert(m->columns() == columns());

	AtlasMatrix* result = new AtlasMatrix(*this);
	
	// TODO: faster
	auto rValue = result->_data.begin();
	for(auto value = m->_data.begin(); value != m->_data.end();
		++value, ++rValue)
	{
		*rValue += *value;
	}
	
	return result;
}

Value* AtlasMatrix::addBroadcastRow(const Value* matrix) const
{
	auto m = dynamic_cast<const AtlasMatrix*>(matrix);	
	assert(m != nullptr);
	
	assert(m->columns() == columns());

	AtlasMatrix* result = new AtlasMatrix(*this);
	
	// TODO: faster
	size_t columnSize = columns();
	size_t rowSize    = rows();

	for(size_t r = 0; r < rowSize; ++r)
	{
		for(size_t c = 0; c < columnSize; ++c)
		{
			result->data()[result->getPosition(r, c)] =
				data()[getPosition(r, c)] + m->data()[m->getPosition(0, c)];
		}
	}

	return result;
}

Value* AtlasMatrix::add(float f) const
{
	AtlasMatrix* result = new AtlasMatrix(*this);
	
	// TODO: faster
	for(auto& value : result->_data)
	{
		value += f;
	}
	
	return result;
}

Value* AtlasMatrix::subtract(const Value* matrix) const
{
	auto m = dynamic_cast<const AtlasMatrix*>(matrix);	
	assert(m != nullptr);
	
	assert(m->rows()    == rows());
	assert(m->columns() == columns());

	AtlasMatrix* result = new AtlasMatrix(*this);
	
	// TODO: faster
	auto rValue = result->_data.begin();
	for(auto value = m->_data.begin(); value != m->_data.end();
		++value, ++rValue)
	{
		*rValue -= *value;
	}
	
	return result;
}

Value* AtlasMatrix::subtract(float f) const
{
	AtlasMatrix* result = new AtlasMatrix(*this);
	
	// TODO: faster
	for(auto& value : result->_data)
	{
		value -= f;
	}
	
	return result;
}

Value* AtlasMatrix::slice(size_t startRow, size_t startColumn,
	size_t rows, size_t columns) const
{
	AtlasMatrix* result = new AtlasMatrix(rows, columns);
	
	assert(startRow    + rows    <= this->rows()   );
	assert(startColumn + columns <= this->columns());
	
	for(size_t row = 0; row != rows; ++row)
	{
		std::memcpy(&result->data()[result->getPosition(row, 0)],
			&data()[getPosition(row + startRow, startColumn)],
			columns * sizeof(float));
	}
	
	return result;
}

Value* AtlasMatrix::log() const
{
    AtlasMatrix* result = new AtlasMatrix(*this);
	
	result->logSelf();

    return result;
}

Value* AtlasMatrix::abs() const
{
    AtlasMatrix* result = new AtlasMatrix(*this);
	
	result->absSelf();

    return result;
}

Value* AtlasMatrix::negate() const
{
    AtlasMatrix* result = new AtlasMatrix(*this);
	
	result->negateSelf();

    return result;
}

Value* AtlasMatrix::sigmoid() const
{
    AtlasMatrix* result = new AtlasMatrix(*this);
	
	result->sigmoidSelf();

    return result;
}

Value* AtlasMatrix::sigmoidDerivative() const
{
    AtlasMatrix* result = new AtlasMatrix(*this);
	
	result->sigmoidDerivativeSelf();

    return result;
}

void AtlasMatrix::negateSelf()
{
	for(auto& f : _data)
	{
		f = -f;
	}
}

void AtlasMatrix::logSelf()
{
	for(auto& f : _data)
	{
		f = std::log(f);
	}
}

void AtlasMatrix::absSelf()
{
	for(auto& f : _data)
	{
		f = std::abs(f);
	}
}

static float sigmoid(float v)
{
    if(v < -50.0f) return 0.0f;
    if(v > 50.0f)  return 1.0f;
    
    return 1.0f / (1.0f + std::exp(-v)); 
}

static float sigmoidDerivative(float v)
{
    // f(x) = 1/(1+e^-x)
    // dy/dx = f(x)' = f(x) * (1 - f(x))
	float element = sigmoid(v) * (1.0f - sigmoid(v));
	
	element = element * (1.0f - element);
	
	return element;
}

void AtlasMatrix::sigmoidSelf()
{
	for(auto& f : _data)
	{
		f = matrix::sigmoid(f);
	}
}

void AtlasMatrix::sigmoidDerivativeSelf()
{
	for(auto& f : _data)
	{
		f = matrix::sigmoidDerivative(f);
	}
}

void AtlasMatrix::assignUniformRandomValues(float min, float max)
{
	std::default_random_engine generator(std::time(0));
	std::uniform_real_distribution<float> distribution(min, max);

	for(auto& f : _data)
	{
		f = distribution(generator);
	}
}

Value* AtlasMatrix::greaterThanOrEqual(float f) const
{
	AtlasMatrix* result = new AtlasMatrix(*this);
	
	// TODO: faster
	for(auto& value : result->_data)
	{
		value = (value >= f) ? 1.0f : 0.0f;
	}
	
	return result;
}

Value* AtlasMatrix::equals(const Value* m) const
{
	assert(m->size() == size());

	AtlasMatrix* result = new AtlasMatrix(*this);
	
	// TODO: faster
	auto value = m->data().begin();
	for(auto resultValue = result->data().begin(); resultValue != result->data().end();
		++resultValue, ++value)
	{
		*resultValue = (*resultValue == *value) ? 1.0f : 0.0f;
	}
	
	return result;
}

void AtlasMatrix::transposeSelf()
{
    // TODO: in place
	auto matrix = transpose();
	
	auto atlasMatrix = dynamic_cast<AtlasMatrix*>(matrix);
	assert(atlasMatrix != nullptr);
	
	*this = *atlasMatrix;
	
	delete atlasMatrix;
}

float AtlasMatrix::reduceSum() const
{
    float sum = 0.0f;

    for(auto& f : _data)
    {
        sum += f;
    }

    return sum;
}

const AtlasMatrix::FloatVector& AtlasMatrix::data() const
{
	return _data;
}

AtlasMatrix::FloatVector& AtlasMatrix::data()
{
	return _data;
}

Value* AtlasMatrix::clone() const
{
	return new AtlasMatrix(*this);
}

bool AtlasMatrix::isSupported()
{
	AtlasLibrary::load();

	return AtlasLibrary::loaded();
}

}

}

