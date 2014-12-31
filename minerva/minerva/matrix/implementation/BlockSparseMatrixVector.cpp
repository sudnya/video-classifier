/*	\file   BlockSparseMatrixVector.cpp
	\date   Sunday August 11, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the BlockSparseMatrixVector class.
*/

// Minerva Includes
#include <minerva/matrix/interface/BlockSparseMatrixVector.h>

// Standard Library Includes
#include <cassert>

namespace minerva
{

namespace matrix
{

BlockSparseMatrixVector::BlockSparseMatrixVector()
{

}

BlockSparseMatrixVector::BlockSparseMatrixVector(size_t elements, const BlockSparseMatrix& value)
: _matrix(elements, value)
{

}

BlockSparseMatrixVector::BlockSparseMatrixVector(const BlockSparseMatrixVector& m)
: _matrix(m._matrix)
{

}

BlockSparseMatrixVector::BlockSparseMatrixVector(const BlockSparseMatrixVector&& m)
: _matrix(std::move(m._matrix))
{

}

BlockSparseMatrixVector& BlockSparseMatrixVector::operator=(const BlockSparseMatrixVector& m)
{
	_matrix = m._matrix;
	
	return *this;
}

BlockSparseMatrixVector& BlockSparseMatrixVector::operator=(const BlockSparseMatrixVector&& m)
{
	_matrix = std::move(m._matrix);
	
	return *this;
}

BlockSparseMatrixVector::reference_type BlockSparseMatrixVector::operator[](size_t i)
{
	return _matrix[i];
}

BlockSparseMatrixVector::const_reference_type BlockSparseMatrixVector::operator[](size_t i) const
{
	return _matrix[i];
}

BlockSparseMatrixVector BlockSparseMatrixVector::negate() const
{
	BlockSparseMatrixVector result;
	
	result.reserve(size());

	// TODO: in parallel
	for(auto& i : *this)
	{
		result.push_back(i.negate());
	}
	
	return result;
}

BlockSparseMatrixVector BlockSparseMatrixVector::subtract(const BlockSparseMatrixVector& m) const
{
	assert(m.size() == size());
	
	BlockSparseMatrixVector result;
	
	result.reserve(size());

	// TODO: in parallel
	for(auto i = begin(), j = m.begin(); i != end(); ++i, ++j)
	{
		result.push_back(i->subtract(*j));
	}
	
	return result;
}

BlockSparseMatrixVector BlockSparseMatrixVector::add(const BlockSparseMatrixVector& m) const
{
	assert(m.size() == size());
	
	BlockSparseMatrixVector result;
	
	result.reserve(size());

	// TODO: in parallel
	for(auto i = begin(), j = m.begin(); i != end(); ++i, ++j)
	{
		result.push_back(i->add(*j));
	}
	
	return result;
}

BlockSparseMatrixVector BlockSparseMatrixVector::elementMultiply(const BlockSparseMatrixVector& m) const
{
	assert(m.size() == size());
	
	BlockSparseMatrixVector result;
	
	result.reserve(size());

	// TODO: in parallel
	for(auto i = begin(), j = m.begin(); i != end(); ++i, ++j)
	{
		result.push_back(i->elementMultiply(*j));
	}
	
	return result;
}

BlockSparseMatrixVector BlockSparseMatrixVector::add(float f) const
{
	BlockSparseMatrixVector result;
	
	result.reserve(size());

	// TODO: in parallel
	for(auto& i : *this)
	{
		result.push_back(i.add(f));
	}
	
	return result;
}

BlockSparseMatrixVector BlockSparseMatrixVector::multiply(float f) const
{
	BlockSparseMatrixVector result;
	
	result.reserve(size());

	// TODO: in parallel
	for(auto& i : *this)
	{
		result.push_back(i.multiply(f));
	}
	
	return result;
}

void BlockSparseMatrixVector::addSelf(const BlockSparseMatrixVector& m)
{
	assert(m.size() == size());
	
	// TODO: in parallel
	auto j = m.begin();
	for(auto i = begin(); i != end(); ++i, ++j)
	{
		// TODO: Add an addSelf method
		*i = i->add(*j);
	}
}

void BlockSparseMatrixVector::multiplySelf(float f)
{
	// TODO: in parallel
	for(auto& i : *this)
	{
		// TODO: Add a multiply self method
		i = i.multiply(f);
	}
}

float BlockSparseMatrixVector::dotProduct(const BlockSparseMatrixVector& m) const
{
	assert(m.size() == size());
	
	float sum = 0.0f;
	
	auto j = m.begin();
	for(auto i = begin(); i != end(); ++i, ++j)
	{
		sum += j->elementMultiply(*i).reduceSum();
	}
	
	return sum;
}

float BlockSparseMatrixVector::reduceSum() const
{
	float sum = 0.0f;
	
	// TODO: in parallel
	for(auto& i : *this)
	{
		sum += i.reduceSum();
	}
	
	return sum;
}

bool BlockSparseMatrixVector::empty() const
{
	return _matrix.empty();
}

size_t BlockSparseMatrixVector::size() const
{
	return _matrix.size();
}

void BlockSparseMatrixVector::reserve(size_t size)
{
	_matrix.reserve(size);
}

void BlockSparseMatrixVector::resize(size_t size)
{
	_matrix.resize(size);
}

BlockSparseMatrixVector::iterator BlockSparseMatrixVector::begin()
{
	return _matrix.begin();
}

BlockSparseMatrixVector::const_iterator BlockSparseMatrixVector::begin() const
{
	return _matrix.begin();
}

BlockSparseMatrixVector::iterator BlockSparseMatrixVector::end()
{
	return _matrix.end();
}

BlockSparseMatrixVector::const_iterator BlockSparseMatrixVector::end() const
{
	return _matrix.end();
}

BlockSparseMatrixVector::reverse_iterator BlockSparseMatrixVector::rbegin()
{
	return _matrix.rbegin();
}

BlockSparseMatrixVector::const_reverse_iterator BlockSparseMatrixVector::rbegin() const
{
	return _matrix.rbegin();
}

BlockSparseMatrixVector::reverse_iterator BlockSparseMatrixVector::rend()
{
	return _matrix.rend();
}

BlockSparseMatrixVector::const_reverse_iterator BlockSparseMatrixVector::rend() const
{
	return _matrix.rend();
}

void BlockSparseMatrixVector::push_back(const BlockSparseMatrix& m)
{
	_matrix.push_back(m);
}

void BlockSparseMatrixVector::push_back(BlockSparseMatrix&& m)
{
	_matrix.push_back(std::move(m));
}

BlockSparseMatrix& BlockSparseMatrixVector::back()
{
	return _matrix.back();
}

const BlockSparseMatrix& BlockSparseMatrixVector::back() const
{
	return _matrix.back();
}

BlockSparseMatrix& BlockSparseMatrixVector::front()
{
	return _matrix.front();
}

const BlockSparseMatrix& BlockSparseMatrixVector::front() const
{
	return _matrix.front();
}

std::string BlockSparseMatrixVector::toString() const
{
	if(empty())
	{
		return "[]";
	}
	
	return front().toString();
}

}

}


