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

Matrix::Matrix(size_t r, size_t c, const FloatVector& d)
: _matrix(MatrixImplementation::createBestImplementation(r, c, d))
{

}

Matrix::~Matrix()
{
	delete _matrix;
}

Matrix::Matrix(const Matrix& m)
: _matrix(nullptr)
{
	if(m._matrix != nullptr)
	{
		_matrix = m._matrix->clone();
	}
}

Matrix::Matrix(Matrix&& m)
: _matrix(nullptr)
{
	std::swap(_matrix, m._matrix);
}
	
Matrix& Matrix::operator=(const Matrix& m)
{
	if(&m == this) return *this;
	
	delete _matrix;
	
	_matrix = nullptr;
	
	if(m._matrix != nullptr)
	{
		_matrix = m._matrix->clone();
	}

	return *this;
}

Matrix& Matrix::operator=(Matrix&& m)
{
	std::swap(_matrix, m._matrix);

	return *this;
}

Matrix::iterator Matrix::begin()
{
	return _matrix->data().begin();
}

Matrix::const_iterator Matrix::begin() const
{
	return _matrix->data().begin();
}

Matrix::iterator Matrix::end()
{
	return _matrix->data().end();
}

Matrix::const_iterator Matrix::end() const
{
	return _matrix->data().end();
}

Matrix::FloatReference Matrix::operator[](size_t index)
{
	return _matrix->data()[index];
}

Matrix::ConstFloatReference Matrix::operator[](size_t index) const
{
	return _matrix->data()[index];
}

Matrix::FloatReference Matrix::operator()(size_t row, size_t column)
{
	size_t position = getPosition(row, column);

	return (*this)[position];
}

Matrix::ConstFloatReference Matrix::operator()(size_t row, size_t column) const
{
	size_t position = getPosition(row, column);

	return (*this)[position];
}

size_t Matrix::size() const
{
	if(_matrix == nullptr)
	{
		return 0;
	}
	
	return _matrix->size();
}

bool Matrix::empty() const
{
    return size() == 0;
}

void Matrix::resize(size_t rows, size_t columns)
{
	assert(_matrix != nullptr);
	
	_matrix->resize(rows, columns);
}

Matrix Matrix::getColumn(size_t number) const
{
	return slice(0, number, rows(), 1);
}

Matrix Matrix::getRow(size_t number) const
{
	return slice(number, 0, 1, columns());
}

size_t Matrix::columns() const
{
	assert(_matrix != nullptr);
	
	return _matrix->columns();
}

size_t Matrix::rows() const
{
	assert(_matrix != nullptr);
	
	return _matrix->rows();
}

size_t Matrix::getPosition(size_t row, size_t column) const
{
	return row * columns() + column;
}

Matrix Matrix::multiply(float f) const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->multiply(f));
}

Matrix Matrix::multiply(const Matrix& m) const
{
	assert(_matrix != nullptr);
	assert(columns() == m.rows());

	return Matrix(_matrix->multiply(m._matrix));
}

Matrix Matrix::elementMultiply(const Matrix& m) const
{
	assert(_matrix != nullptr);
	assert(m.rows()    == rows()   );
	assert(m.columns() == columns());

	return Matrix(_matrix->elementMultiply(m._matrix));
}

Matrix Matrix::add(const Matrix& m) const
{
	assert(_matrix != nullptr);
	
	assert(m.rows()    == rows());
	assert(m.columns() == columns());

	return Matrix(_matrix->add(m._matrix));
}

Matrix Matrix::addBroadcastRow(const Matrix& m) const
{
	assert(_matrix != nullptr);
	
	assert(m.columns() == columns());

	return Matrix(_matrix->addBroadcastRow(m._matrix));
}

Matrix Matrix::add(float f) const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->add(f));
}

Matrix Matrix::subtract(const Matrix& m) const
{
	assert(_matrix != nullptr);
	
	assert(m.rows()    == rows());
	assert(m.columns() == columns());

	return Matrix(_matrix->subtract(m._matrix));
}

Matrix Matrix::subtract(float f) const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->subtract(f));
}

Matrix Matrix::slice(size_t startRow, size_t startColumn,
	size_t rows, size_t columns) const
{
	assert(_matrix != nullptr);
	
	assert(startRow    + rows    <= this->rows()   );
	assert(startColumn + columns <= this->columns());
	
	return Matrix(_matrix->slice(startRow, startColumn, rows, columns));
}

Matrix Matrix::transpose() const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->transpose());
}

Matrix Matrix::appendColumns(const Matrix& m) const
{
	assert(_matrix != nullptr);
	
	assert(empty() || (rows() == m.rows()));

	return Matrix(_matrix->appendColumns(m._matrix));
}

Matrix Matrix::appendRows(const Matrix& m) const
{
	assert(_matrix != nullptr);
	
	assert(empty() || (columns() == m.columns()));

	return Matrix(_matrix->appendRows(m._matrix));
}

Matrix Matrix::log() const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->log());
}

Matrix Matrix::abs() const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->abs());
}

Matrix Matrix::negate() const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->negate());
}

Matrix Matrix::sigmoid() const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->sigmoid());
}

Matrix Matrix::sigmoidDerivative() const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->sigmoidDerivative());
}

Matrix Matrix::klDivergence(float sparsity) const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->klDivergence(sparsity));
}

Matrix Matrix::klDivergenceDerivative(float sparsity) const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->klDivergenceDerivative(sparsity));
}

void Matrix::negateSelf()
{
	assert(_matrix != nullptr);

	_matrix->negateSelf();
}

void Matrix::logSelf()
{
	assert(_matrix != nullptr);

	_matrix->logSelf();
}

void Matrix::sigmoidSelf()
{
	assert(_matrix != nullptr);

	_matrix->sigmoidSelf();
}

void Matrix::sigmoidDerivativeSelf()
{
	assert(_matrix != nullptr);

	_matrix->sigmoidDerivativeSelf();
}

void Matrix::klDivergenceSelf(float sparsity)
{
	assert(_matrix != nullptr);

	_matrix->klDivergenceSelf(sparsity);
}

void Matrix::klDivergenceDerivativeSelf(float sparsity)
{
	assert(_matrix != nullptr);

	_matrix->klDivergenceDerivativeSelf(sparsity);
}

void Matrix::minSelf(float f)
{
	assert(_matrix != nullptr);
	
	_matrix->minSelf(f);
}

void Matrix::maxSelf(float f)
{
	assert(_matrix != nullptr);
	
	_matrix->maxSelf(f);
}

void Matrix::assignUniformRandomValues(
	std::default_random_engine& engine, float min, float max)
{
	assert(_matrix != nullptr);

	_matrix->assignUniformRandomValues(engine, min, max);
}

Matrix Matrix::greaterThanOrEqual(float f) const
{
	assert(_matrix != nullptr);

	return Matrix(_matrix->greaterThanOrEqual(f));
}

Matrix Matrix::equals(const Matrix& m) const
{
	assert(_matrix != nullptr);

	return Matrix(_matrix->equals(m._matrix));
}

Matrix Matrix::lessThanOrEqual(float f) const
{
	assert(_matrix != nullptr);

	return Matrix(_matrix->lessThanOrEqual(f));
}

void Matrix::transposeSelf()
{
	assert(_matrix != nullptr);

	_matrix->transposeSelf();
}

float Matrix::reduceSum() const
{
	assert(_matrix != nullptr);

	return _matrix->reduceSum();
}

Matrix Matrix::reduceSumAlongColumns() const
{
	assert(_matrix != nullptr);

	return Matrix(_matrix->reduceSumAlongColumns());
}

Matrix Matrix::reduceSumAlongRows() const
{
	// TODO implement this
	return transpose().reduceSumAlongColumns().transpose();
}

void Matrix::clear()
{
	assert(_matrix != nullptr);
	
	_matrix->resize(0, 0);
}

const Matrix::FloatVector& Matrix::data() const
{
	assert(_matrix != nullptr);

	return _matrix->data();
}

Matrix::FloatVector& Matrix::data()
{
	assert(_matrix != nullptr);

	return _matrix->data();
}

Matrix::Matrix(MatrixImplementation* i)
: _matrix(i)
{

}

bool Matrix::operator==(const Matrix& m) const
{
	return data() == m.data();
}

bool Matrix::operator!=(const Matrix& m) const
{
	return data() != m.data();
}

std::string Matrix::toString(size_t maxRows, size_t maxColumns) const
{
    std::stringstream stream;

	stream << shapeString() << " ";

    stream << "[ ";

	size_t finalRow = std::min(rows(), maxRows);

    for(size_t row = 0; row != finalRow; ++row)
    {
		size_t finalColumn = std::min(columns(), maxColumns);

        for(size_t column = 0; column != finalColumn; ++column)
        {
            stream << (*this)(row, column) << " ";
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
    std::stringstream stream;
	
	stream << "(" << rows() << " rows, " << columns() << " columns)";

    return stream.str();
}

}

}


