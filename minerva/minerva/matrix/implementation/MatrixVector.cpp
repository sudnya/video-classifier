/*	\file   MatrixVector.cpp
	\date   Sunday August 11, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the MatrixVector class.
*/

// Minerva Includes
#include <minerva/matrix/interface/MatrixVector.h>

// Standard Library Includes
#include <cassert>

namespace minerva
{

namespace matrix
{

MatrixVector::MatrixVector()
{

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

void MatrixVector::reserve(size_t size)
{
	_matrix.reserve(size);
}

void MatrixVector::resize(size_t size)
{
	_matrix.resize(size);
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

}

}


