/*	\file   MatrixImplementation.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the MatrixImplementation class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <cstring>
#include <random>

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
	MatrixImplementation(size_t rows, size_t columns, const FloatVector& v);
	virtual ~MatrixImplementation();

public:
	inline size_t size()  const;
	inline bool   empty() const;

    inline size_t columns() const;
	inline size_t rows()	const;
	
	inline size_t getPosition(size_t row, size_t columns) const;
	
public:
	virtual void resize(size_t rows, size_t columns) = 0;

public:
	virtual Value* appendColumns(const Value* m) const = 0;
	virtual Value* appendRows(const Value* m) const = 0;
	virtual Value* transpose() const = 0;
 
public: 
	virtual Value* multiply(const Value* m) const = 0;
	virtual Value* multiply(float f) const = 0;
	virtual Value* elementMultiply(const Value* m) const = 0;

	virtual Value* add(const Value* m) const = 0;
	virtual Value* addBroadcastRow(const Value* m) const = 0;
	virtual Value* add(float f) const = 0;

	virtual Value* subtract(const Value* m) const = 0;
	virtual Value* subtract(float f) const = 0;

	virtual Value* log() const = 0;
	virtual Value* abs() const = 0;
	virtual Value* negate() const = 0;
	virtual Value* sigmoid() const = 0;
	virtual Value* sigmoidDerivative() const = 0;
	virtual Value* klDivergence(float sparsity) const = 0;
	virtual Value* klDivergenceDerivative(float sparsity) const = 0;

public:
	virtual Value* slice(size_t startRow, size_t startColumn,
		size_t rows, size_t columns) const = 0;

public:
	virtual void negateSelf() = 0;
	virtual void logSelf() = 0;
	virtual void absSelf() = 0;
    virtual void sigmoidSelf() = 0;
    virtual void sigmoidDerivativeSelf() = 0;
    virtual void klDivergenceSelf(float sparsity) = 0;
    virtual void klDivergenceDerivativeSelf(float sparsity) = 0;
    virtual void minSelf(float f) = 0;
    virtual void maxSelf(float f) = 0;

	virtual void assignUniformRandomValues(
		std::default_random_engine& engine, float min, float max) = 0;
	virtual void transposeSelf() = 0;

public:
	virtual Value* greaterThanOrEqual(float f) const = 0;
	virtual Value* equals(const Value* m) const = 0;
	virtual Value* lessThanOrEqual(float f) const = 0;

public:
    virtual float  reduceSum() const = 0;
    virtual Value* reduceSumAlongColumns() const = 0;

public:
	virtual FloatVector& data() = 0;
	virtual const FloatVector& data() const = 0;

public:
	virtual Value* clone() const = 0;

public:
	static Value* createBestImplementation(size_t rows,
		size_t columns, const FloatVector& f);

protected:
	FloatVector _data;

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

inline size_t MatrixImplementation::getPosition(size_t row, size_t column) const
{
	return row * columns() + column;
}

}

}




