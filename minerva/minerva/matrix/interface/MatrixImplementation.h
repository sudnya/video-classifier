/*	\file   MatrixImplementation.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the MatrixImplementation class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <cstring>

namespace minerva
{

namespace matrix
{

class MatrixImplementation
{
public:
	typedef std::vector<float> FloatVector;
	typedef MatrixImplementation Value;

public:
	MatrixImplementation(size_t rows, size_t columns);
	virtual ~MatrixImplementation();

public:
	inline size_t size()  const;
	inline bool   empty() const;

    inline size_t columns() const;
	inline size_t rows()	const;
	
public:
	virtual void resize(size_t rows, size_t columns) = 0;

public:
	virtual Value* append(const Value* m) const = 0;
	virtual Value* transpose() const = 0;
 
public: 
	virtual Value* multiply(const Value* m) const = 0;
	virtual Value* multiply(float f) const = 0;
	virtual Value* elementMultiply(const Value* m) const = 0;

	virtual Value* add(const Value* m) const = 0;
	virtual Value* add(float f) const = 0;

	virtual Value* subtract(const Value* m) const = 0;
	virtual Value* subtract(float f) const = 0;

	virtual Value* log() const = 0;
	virtual Value* negate() const = 0;
	virtual Value* sigmoid() const = 0;

public:
	virtual Value* slice(size_t startRow, size_t startColumn,
		size_t rows, size_t columns) const = 0;

public:
	virtual void negateSelf() = 0;
	virtual void logSelf() = 0;
    virtual void sigmoidSelf() = 0;

	virtual void transposeSelf() = 0;

public:
    virtual float reduceSum() const = 0;

public:
	virtual FloatVector data() const = 0;
	virtual void setDataRowMajor(const FloatVector& data) = 0;
	
public:
	virtual void  setValue(size_t row, size_t column, float value) = 0;
	virtual float getValue(size_t row, size_t column) const = 0;

public:
	virtual Value* clone() const = 0;

public:
	static Value* createBestImplementation(size_t rows,
		size_t columns, const FloatVector& f);

protected:
	size_t _rows;
	size_t _columns;

};

inline size_t MatrixImplementation::size()  const
{
	return rows() * columns();
}

inline bool MatrixImplementation::empty() const
{
	return rows() == 0 || columns() == 0;
}

inline size_t MatrixImplementation::columns() const
{
	return _columns;
}

inline size_t MatrixImplementation::rows()	const
{
	return _rows;
}

}

}




