/*	\file   Matrix.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Matrix class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <cstddef>

// Forward Declarations
namespace minerva { namespace matrix { class Reference;      } }
namespace minerva { namespace matrix { class ConstReference; } }
namespace minerva { namespace matrix { class Pointer;        } }
namespace minerva { namespace matrix { class ConstPointer;   } }
namespace minerva { namespace matrix { class Dimension;      } }
namespace minerva { namespace matrix { class Matrix;         } }

namespace minerva
{

namespace matrix
{

/*! \brief An interface to operations on a general purpose array. */
class Matrix
{
public:
	typedef      Pointer       iterator;
	typedef ConstPointer const_iterator;

public:
	Matrix();
	Matrix(const Dimension& size);
	Matrix(const Dimension& size, const Dimension& stride);
	Matrix(const Dimension& size, const Dimension& stride, const Precision& precision);
	Matrix(const Dimension& size, const Dimension& stride, const Precision& precision, const Allocation& allocation);

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
	void clear();

public:
	Allocation allocation();

public:
    std::string toString(size_t maxRows = 20, size_t maxColumns = 20) const;
	std::string debugString() const;
	std::string shapeString() const;

public:
	template<typename... Args>
	FloatReference operator()(Args... args)
	{
		return (*this)[Dimension(args)];
	}

	template<typename... Args>
	ConstFloatReference operator()(Args... args) const
	{
		return (*this)[Dimension[args]];
	}

public:
	FloatReference      operator[](const Dimension& d);
	ConstFloatReference operator[](const Dimension& d) const;

private:
	Allocation _allocation;

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


