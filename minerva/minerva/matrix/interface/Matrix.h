/*	\file   Matrix.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Matrix class.
*/

#pragma once

// Minerva Includes
#include <minerva/matrix/interface/Vector.h>

// Forward Declarations
namespace minerva { namespace matrix { class MatrixImplementation; } }

namespace minerva
{

namespace matrix
{

class Matrix
{
public:
	typedef std::vector<float>		  FloatVector;
	typedef FloatVector::iterator	   iterator;
	typedef FloatVector::const_iterator const_iterator;

public:
	Matrix(size_t rows = 0, size_t colums = 0,
		const FloatVector& data = FloatVector());

public:
	iterator	   begin();
	const_iterator begin() const;

	iterator	   end();
	const_iterator end() const;

public:
	      float& operator[](size_t index);
	const float& operator[](size_t index) const;

	      float& operator()(size_t row, size_t column);
	const float& operator()(size_t row, size_t column) const;

public:
	Vector getColumn(size_t number) const;
	Vector getRow(size_t number) const;
	
public:
	Matrix append(const Matrix& m) const;
	void resize(size_t rows, size_t columns);
	
public:
	size_t size()  const;
	bool   empty() const;

    size_t columns() const;
	size_t rows()	const;
 
 	size_t getPosition(size_t row, size_t column) const;
 
public: 
	Matrix multiply(const Matrix& m) const;
	Matrix multiply(float f) const;
	Matrix elementMultiply(const Matrix& m) const;

	Matrix add(const Matrix& m) const;
	Matrix add(float f) const;

	Matrix subtract(const Matrix& m) const;

	Matrix log() const;
	Matrix negate() const;
	Matrix sigmoid() const;

public:
	Matrix slice(size_t startRow, size_t startColumn,
		size_t rows, size_t columns) const;
	Matrix transpose() const;

public:	
	void appendRowData(const FloatVector& f);
    void setRowData(size_t row, const FloatVector& f);

public:
	void negateSelf();
	void logSelf();
    void sigmoidSelf();

	void transposeSelf();
	
public:
	void* data();
	const void* data() const;

public:
    std::string toString(size_t maxRows = 10, size_t maxColumns = 10) const;

private:
	size_t _rows;
	size_t _columns;
	
	FloatVector _data;
	
private:
	MatrixImplementation* _matrix;

};

}

}


