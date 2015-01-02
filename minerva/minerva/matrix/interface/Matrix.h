/*	\file   Matrix.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Matrix class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <cstddef>
#include <random>

// Forward Declarations
namespace minerva { namespace matrix { class MatrixImplementation; } }

namespace minerva
{

namespace matrix
{

class Matrix
{
public:
	typedef std::vector<float> FloatVector;

public:
	typedef       float& FloatReference;
	typedef const float& ConstFloatReference;
	typedef       float* FloatPointer;
	typedef const float* ConstFloatPointer;

	typedef FloatVector::iterator       iterator;
	typedef FloatVector::const_iterator const_iterator;

public:
	explicit Matrix(size_t rows = 0, size_t colums = 0,
		const FloatVector& data = FloatVector());
	explicit Matrix(Matrix&&);
	~Matrix();

public:
	Matrix(const Matrix&);
	Matrix& operator=(const Matrix&);
	Matrix& operator=(Matrix&&);

public:
	iterator	   begin();
	const_iterator begin() const;

	iterator	   end();
	const_iterator end() const;

public:
	     FloatReference operator[](size_t index);
	ConstFloatReference operator[](size_t index) const;

	     FloatReference operator()(size_t row, size_t column);
	ConstFloatReference operator()(size_t row, size_t column) const;

public:
	Matrix getColumn(size_t number) const;
	Matrix getRow(size_t number) const;
	
public:
	Matrix appendColumns(const Matrix& m) const;
	Matrix appendRows(const Matrix& m) const;
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
	Matrix addBroadcastRow(const Matrix& m) const;
	Matrix add(float f) const;

	Matrix subtract(const Matrix& m) const;
	Matrix subtract(float f) const;

	Matrix log() const;
	Matrix sqrt() const;
	Matrix abs() const;
	Matrix negate() const;
	
	Matrix sigmoid() const;
	Matrix sigmoidDerivative() const;

	Matrix rectifiedLinear() const;
	Matrix rectifiedLinearDerivative() const;

	Matrix klDivergence(float sparsity) const;
	Matrix klDivergenceDerivative(float sparsity) const;

public:
	Matrix slice(size_t startRow, size_t startColumn,
		size_t rows, size_t columns) const;
	Matrix transpose() const;

public:
	void negateSelf();
	void logSelf();
    void sigmoidSelf();
    void sigmoidDerivativeSelf();
    void rectifiedLinearSelf();
    void rectifiedLinearDerivativeSelf();
    void klDivergenceSelf(float sparsity);
    void klDivergenceDerivativeSelf(float sparsity);
	void minSelf(float f);
	void maxSelf(float f);
	void assignSelf(float f);

	void transposeSelf();

	void assignUniformRandomValues(
		std::default_random_engine& engine, float min, float max);

public:
	Matrix greaterThanOrEqual(float f) const;
	Matrix equals(const Matrix& m) const;
	Matrix lessThanOrEqual(float f) const;

public:
    float  reduceSum()             const;
	Matrix reduceSumAlongColumns() const;
	Matrix reduceSumAlongRows()    const;

public:
	void clear();

public:
	const FloatVector& data() const;
	FloatVector& data();

public:
	bool operator==(const Matrix& m) const;
	bool operator!=(const Matrix& m) const;

public:
    std::string toString(size_t maxRows = 20, size_t maxColumns = 20) const;
	std::string debugString() const;
	std::string shapeString() const;

private:
	Matrix(MatrixImplementation* implementation);

private:
	MatrixImplementation* _matrix;

};

}

}


