/*	\file   NaiveMatrix.h
	\date   Monday September 2, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NaiveMatrix class.
*/

#pragma once

// Minerva Includes
#include <minerva/matrix/interface/MatrixImplementation.h>

namespace minerva
{

namespace matrix
{

class NaiveMatrix : public MatrixImplementation
{
public:
	NaiveMatrix(size_t rows, size_t columns,
		const FloatVector& data = FloatVector());
	
public:
	virtual void resize(size_t rows, size_t columns);

public:
	virtual Value* appendColumns(const Value* m) const;
	virtual Value* appendRows(const Value* m) const;
	virtual Value* transpose() const;
 
public: 
	virtual Value* multiply(const Value* m) const;
	virtual Value* multiply(float f) const;
	virtual Value* elementMultiply(const Value* m) const;

	virtual Value* add(const Value* m) const;
	virtual Value* add(float f) const;

	virtual Value* subtract(const Value* m) const;
	virtual Value* subtract(float f) const;

	virtual Value* log() const;
	virtual Value* abs() const;
	virtual Value* negate() const;
	virtual Value* sigmoid() const;
	virtual Value* sigmoidDerivative() const;

public:
	virtual Value* slice(size_t startRow, size_t startColumn,
		size_t rows, size_t columns) const;

public:
	virtual void negateSelf();
	virtual void logSelf();
	virtual void absSelf();
    virtual void sigmoidSelf();
    virtual void sigmoidDerivativeSelf();

	virtual void assignUniformRandomValues(float min, float max);
	virtual void transposeSelf();

public:
	virtual Value* greaterThanOrEqual(float f) const;
	virtual Value* equals(const Value* m) const;

public:
    virtual float reduceSum() const;

public:
	virtual const FloatVector& data() const;
	virtual FloatVector& data();

public:
	virtual Value* clone() const;

private:
	FloatVector _data;

};

}

}




