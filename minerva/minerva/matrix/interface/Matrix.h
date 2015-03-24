/*	\file   Matrix.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Matrix class.
*/

#pragma once

// Minerva Includes
#include <minerva/matrix/interface/FloatReference.h>
#include <minerva/matrix/interface/FloatIterator.h>
#include <minerva/matrix/interface/Dimension.h>
#include <minerva/matrix/interface/Precision.h>

// Standard Library Includes
#include <string>
#include <cstddef>
#include <memory>

// Forward Declarations
namespace minerva { namespace matrix { class Allocation; } }

namespace minerva
{

namespace matrix
{

/*! \brief An interface to operations on a general purpose array. */
class Matrix
{
public:
	Matrix();
	Matrix(const Dimension& size);
	Matrix(const Dimension& size, const Dimension& stride);
	Matrix(const Dimension& size, const Dimension& stride, const Precision& precision);
	Matrix(const Dimension& size, const Dimension& stride, const Precision& precision, const std::shared_ptr<Allocation>& allocation);

public:
	~Matrix();
	
public:
	const Dimension& size()   const;
	const Dimension& stride() const;

public:
	const Precision& precision() const;

public:
	size_t elements() const;

public:
	FloatIterator begin();
	FloatIterator end();
	
	ConstFloatIterator begin() const;
	ConstFloatIterator end()   const;

public:
	std::shared_ptr<Allocation> allocation();

public:
    std::string toString() const;
	std::string debugString() const;
	std::string shapeString() const;

public:
	template<typename... Args>
	FloatReference operator()(Args... args)
	{
		return (*this)[Dimension(args...)];
	}

	template<typename... Args>
	ConstFloatReference operator()(Args... args) const
	{
		return (*this)[Dimension(args...)];
	}

public:
	FloatReference      operator[](const Dimension& d);
	ConstFloatReference operator[](const Dimension& d) const;

private:
	std::shared_ptr<Allocation> _allocation;

private:
	void* _data_begin;
	
private:
	Dimension _size;
	Dimension _stride;

private:
	Precision _precision;

};

}

}


