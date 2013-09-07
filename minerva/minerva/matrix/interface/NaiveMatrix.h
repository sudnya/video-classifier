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
	virtual Value* negate() const;
	virtual Value* sigmoid() const;

public:
	virtual Value* slice(size_t startRow, size_t startColumn,
		size_t rows, size_t columns) const;

public:
	virtual void negateSelf();
	virtual void logSelf();
    virtual void sigmoidSelf();

	virtual void transposeSelf();

public:
    virtual float reduceSum() const;

public:
	virtual FloatVector data() const;
	virtual void setDataRowMajor(const FloatVector& data);

public:
	virtual void  setValue(size_t row, size_t column, float value);
	virtual float getValue(size_t row, size_t column) const;

public:
	virtual Value* clone() const;

private:
	size_t _getPosition(size_t row, size_t column) const;

private:
	FloatVector _data;

};

}

}




