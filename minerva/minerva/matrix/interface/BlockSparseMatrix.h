/*	\file   BlockSparseMatrix.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the BlockSparseMatrix class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <cstdint>
#include <random>

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }

namespace minerva
{

namespace matrix
{

class BlockSparseMatrix
{
public:
	typedef std::vector<Matrix> MatrixVector;
	typedef std::vector<float> FloatVector;

	typedef MatrixVector::iterator       iterator;
	typedef MatrixVector::const_iterator const_iterator;

public:
	explicit BlockSparseMatrix(size_t blocks = 0, size_t rows = 0,
		size_t columns = 0, bool rowSparse = true);
	explicit BlockSparseMatrix(bool rowSparse);

public:
	iterator	   begin();
	const_iterator begin() const;

	iterator	   end();
	const_iterator end() const;

public:
	      Matrix& front();
	const Matrix& front() const;

	      Matrix& back();
	const Matrix& back() const;

public:
	const Matrix& operator[](size_t position) const;
	      Matrix& operator[](size_t position);

public:
	void pop_back();
	void push_back(const Matrix& m);

public:
	size_t size()   const;
	size_t blocks() const;
	bool   empty()  const;

    size_t columns() const;
	size_t rows()	const;

	bool isRowSparse()    const;
	bool isColumnSparse() const;

public:
	void resize(size_t blocks, size_t rowsPerBlock, size_t columnsPerBlock);
	void setColumnSparse();
	void setRowSparse();

public: 
	BlockSparseMatrix multiply(const BlockSparseMatrix& m) const;
	BlockSparseMatrix multiply(float f) const;
	BlockSparseMatrix elementMultiply(const BlockSparseMatrix& m) const;

	BlockSparseMatrix add(const BlockSparseMatrix& m) const;
	BlockSparseMatrix addBroadcastRow(const BlockSparseMatrix& m) const;
	BlockSparseMatrix add(float f) const;

	BlockSparseMatrix subtract(const BlockSparseMatrix& m) const;
	BlockSparseMatrix subtract(float f) const;

	BlockSparseMatrix log() const;
	BlockSparseMatrix negate() const;
	BlockSparseMatrix sigmoid() const;
	BlockSparseMatrix sigmoidDerivative() const;
	
	BlockSparseMatrix klDivergence(float sparsity) const;
	BlockSparseMatrix klDivergenceDerivative(float sparsity) const;

public:
	BlockSparseMatrix transpose() const;

public:
	void negateSelf();
	void logSelf();
    void sigmoidSelf();
    void sigmoidDerivativeSelf();

	void transposeSelf();

	void assignUniformRandomValues(std::default_random_engine& engine,
		float min = 0.0f, float max = 1.0f);

public:
	BlockSparseMatrix greaterThanOrEqual(float f) const;
	BlockSparseMatrix equals(const BlockSparseMatrix& m) const;

public:
	Matrix toMatrix() const;

public:
    float reduceSum() const;
	BlockSparseMatrix reduceSumAlongColumns() const;
	BlockSparseMatrix reduceSumAlongRows() const;

public:
	std::string toString()    const;
	std::string debugString() const;

private:
	MatrixVector _matrices;
	bool         _isRowSparse;

};

}

}


