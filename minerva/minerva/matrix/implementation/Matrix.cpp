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

Matrix::iterator Matrix::begin()
{
	return iterator(this);
}

Matrix::const_iterator Matrix::begin() const
{
	return const_iterator(this);
}

Matrix::iterator Matrix::end()
{
	return iterator(this, size());
}

Matrix::const_iterator Matrix::end() const
{
	return const_iterator(this, size());
}

Matrix::FloatReference Matrix::operator[](size_t index)
{
	return FloatReference(this, index);
}

Matrix::ConstFloatReference Matrix::operator[](size_t index) const
{
	return ConstFloatReference(this, index);
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

void Matrix::setValue(size_t position, float value)
{
	assert(_matrix != nullptr);

	_matrix->setValue(_getRow(position), _getColumn(position), value);
}

float Matrix::getValue(size_t position) const
{
	assert(_matrix != nullptr);

	return _matrix->getValue(_getRow(position), _getColumn(position));
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

Matrix Matrix::add(float f) const
{
	assert(_matrix != nullptr);
	
	return Matrix(_matrix->add(f));
}

Matrix Matrix::add(const Matrix& m) const
{
	assert(_matrix != nullptr);
	
	assert(m.rows()    == rows());
	assert(m.columns() == columns());

	return Matrix(_matrix->add(m._matrix));
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

Matrix::FloatVector Matrix::data() const
{
	assert(_matrix != nullptr);

	return _matrix->data();
}

void Matrix::setDataRowMajor(const FloatVector& data)
{
	assert(_matrix != nullptr);

	return _matrix->setDataRowMajor(data);
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

    stream << "[ ";

    for(size_t row = 0; row != std::min(rows(), maxRows); ++row)
    {
        for(size_t column = 0;
        	column != std::min(columns(), maxColumns); ++column)
        {
            stream << (*this)(row, column) << " ";
        }
        
        stream << "\n ";
    }

    stream << "]\n";

    return stream.str();
}

size_t Matrix::_getRow(size_t position) const
{
	return position / columns();
}

size_t Matrix::_getColumn(size_t position) const
{
	return position % columns();
}

typedef Matrix::FloatReference      FloatReference;
typedef Matrix::ConstFloatReference ConstFloatReference;

typedef Matrix::FloatPointer      FloatPointer;
typedef Matrix::ConstFloatPointer ConstFloatPointer;

typedef Matrix::iterator       iterator;
typedef Matrix::const_iterator const_iterator;

FloatPointer::FloatPointer(Matrix* matrix, size_t position)
: _matrix(matrix), _position(position)
{

}
	
FloatReference FloatPointer::operator*()
{
	return FloatReference(_matrix, _position);
}

ConstFloatReference FloatPointer::operator*() const
{
	return ConstFloatReference(_matrix, _position);
}

ConstFloatPointer::ConstFloatPointer(const Matrix* matrix, size_t position)
: _matrix(matrix), _position(position)
{

}

ConstFloatPointer::ConstFloatPointer(const FloatPointer& p)
: _matrix(p._matrix), _position(p._position)
{

}

ConstFloatReference ConstFloatPointer::operator*() const
{
	return ConstFloatReference(_matrix, _position);
}

FloatReference::FloatReference(Matrix* matrix, size_t position)
: _matrix(matrix), _position(position)
{

}

FloatReference& FloatReference::operator=(float f)
{
	_matrix->setValue(_position, f);

	return *this;
}

FloatReference& FloatReference::operator+=(float f)
{
	_matrix->setValue(_position, _matrix->getValue(_position) + f);

	return *this;
}

FloatReference& FloatReference::operator-=(float f)
{
	_matrix->setValue(_position, _matrix->getValue(_position) - f);

	return *this;
}

FloatReference::operator float() const
{
	return _matrix->getValue(_position);
}

FloatPointer FloatReference::operator&()
{
	return FloatPointer(_matrix, _position);
}

FloatPointer FloatReference::operator->()
{
	return FloatPointer(_matrix, _position);
}

ConstFloatReference::ConstFloatReference(const Matrix* matrix, size_t position)
: _matrix(matrix), _position(position)
{

}

ConstFloatReference::operator float() const
{
	return _matrix->getValue(_position);
}

ConstFloatPointer ConstFloatReference::operator&()
{
	return ConstFloatPointer(_matrix, _position);
}

ConstFloatPointer ConstFloatReference::operator->()
{
	return ConstFloatPointer(_matrix, _position);
}

iterator::iterator(Matrix* matrix)
: _matrix(matrix), _position(0)
{

}

iterator::iterator(Matrix* matrix, size_t position)
: _matrix(matrix), _position(position)
{

}

FloatReference iterator::operator*()
{
	return FloatReference(_matrix, _position);
}

ConstFloatReference iterator::operator*() const
{
	return ConstFloatReference(_matrix, _position);
}

FloatPointer iterator::operator->()
{
	return FloatPointer(_matrix, _position);
}

ConstFloatPointer iterator::operator->() const
{
	return ConstFloatPointer(_matrix, _position);
}

iterator& iterator::operator++()
{
	++_position;

	return *this;
}

iterator iterator::operator++(int)
{
	iterator temp = *this;

	++(*this);

	return temp;
}

iterator& iterator::operator--()
{
	--_position;
	
	return *this;
}

iterator iterator::operator--(int)
{
	iterator temp = *this;

	--(*this);

	return temp;
}

iterator::difference_type iterator::operator-(const const_iterator& i) const
{
	return _position - i._position;
}

iterator::difference_type iterator::operator-(const Matrix::iterator& i) const
{
	return _position - i._position;
}

iterator::operator const_iterator() const
{
	return const_iterator(_matrix, _position);
}

bool iterator::operator!=(const const_iterator& i) const
{
	return !(*this == i);
}

bool iterator::operator==(const const_iterator& i) const
{
	return i._matrix == _matrix && i._position == _position;
}

bool iterator::operator<(const const_iterator& i) const
{
	if(_matrix < i._matrix) return true;
	
	if(_matrix > i._matrix) return false;
	
	return _position < i._position;
}

bool iterator::operator!=(const Matrix::iterator& i) const
{
	return !(*this == i);
}

bool iterator::operator==(const Matrix::iterator& i) const
{
	return i._matrix == _matrix && i._position == _position;
}

bool iterator::operator<(const Matrix::iterator& i) const
{
	if(_matrix < i._matrix) return true;
	
	if(_matrix > i._matrix) return false;
	
	return _position < i._position;
}

const_iterator::const_iterator(const Matrix* matrix)
: _matrix(matrix), _position(0)
{

}

const_iterator::const_iterator(const Matrix* matrix, size_t position)
: _matrix(matrix), _position(position)
{

}

const_iterator::const_iterator(const Matrix::iterator& i)
: _matrix(i._matrix), _position(i._position)
{

}

ConstFloatReference const_iterator::operator*() const
{
	return ConstFloatReference(_matrix, _position);
}

ConstFloatPointer const_iterator::operator->() const
{
	return ConstFloatPointer(_matrix, _position);
}

const_iterator& const_iterator::operator++()
{
	++_position;

	return *this;
}

const_iterator const_iterator::operator++(int)
{
	const_iterator temp = *this;

	++(*this);

	return temp;
}

const_iterator& const_iterator::operator--()
{
	--_position;
	
	return *this;
}

const_iterator const_iterator::operator--(int)
{
	const_iterator temp = *this;

	--(*this);

	return temp;
}

const_iterator::difference_type const_iterator::operator-(
	const const_iterator& i) const
{
	return _position - i._position;
}

const_iterator::difference_type const_iterator::operator-(
    const Matrix::iterator& i) const
{
	return _position - i._position;
}

bool const_iterator::operator!=(const const_iterator& i) const
{
	return !(*this == i);
}

bool const_iterator::operator==(const const_iterator& i) const
{
	return i._matrix == _matrix && i._position == _position;
}

bool const_iterator::operator<(const const_iterator& i) const
{
	if(_matrix < i._matrix) return true;
	
	if(_matrix > i._matrix) return false;
	
	return _position < i._position;
}

bool const_iterator::operator!=(const Matrix::iterator& i) const
{
	return !(*this == i);
}

bool const_iterator::operator==(const Matrix::iterator& i) const
{
	return i._matrix == _matrix && i._position == _position;
}

bool const_iterator::operator<(const Matrix::iterator& i) const
{
	if(_matrix < i._matrix) return true;
	
	if(_matrix > i._matrix) return false;
	
	return _position < i._position;
}

}

}


