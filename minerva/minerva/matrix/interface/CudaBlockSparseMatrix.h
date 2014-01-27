/*! \file   CudaBlockSparseMatrix.h
	\author Gregory Diamos
	\date   Sunday December 29, 2013
	\brief  The header file for the CudaBlockSparseMatrix class.
*/

#pragma once

// Minerva Includes
#include <minerva/matrix/interface/BlockSparseMatrixImplementation.h>

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace minerva { namespace matrix { class CudaBlockSparseCache; } }

namespace minerva
{

namespace matrix
{

/*! \brief A GPU accelerated implementation of block sparse matrix */
class CudaBlockSparseMatrix : public BlockSparseMatrixImplementation
{
public:
	explicit CudaBlockSparseMatrix(size_t blocks, size_t rows,
		size_t columns, bool rowSparse);
	explicit CudaBlockSparseMatrix(const CudaBlockSparseMatrix&, bool copyData);
	explicit CudaBlockSparseMatrix(const CudaBlockSparseMatrix&);

public:
	virtual ~CudaBlockSparseMatrix();

public: 
	virtual Value* multiply(const Value* m) const;
	virtual Value* convolutionalMultiply(const Value* m, size_t step) const;
	virtual Value* multiply(float f) const;
	virtual Value* elementMultiply(const Value* m) const;

	virtual Value* add(const Value* m) const;
	virtual Value* addBroadcastRow(const Value* m) const;
	virtual Value* convolutionalAddBroadcastRow(const Value* m, size_t step) const;
	virtual Value* add(float f) const;

	virtual Value* subtract(const Value* m) const;
	virtual Value* subtract(float f) const;

	virtual Value* log() const;
	virtual Value* negate() const;
	virtual Value* sigmoid() const;
	virtual Value* sigmoidDerivative() const;
	
	virtual Value* klDivergence(float sparsity) const;
	virtual Value* klDivergenceDerivative(float sparsity) const;

public:
	virtual Value* transpose() const;

public:
	virtual void negateSelf();
	virtual void logSelf();
    virtual void sigmoidSelf();
    virtual void sigmoidDerivativeSelf();

	virtual void transposeSelf();

	virtual void assignUniformRandomValues(std::default_random_engine& engine,
		float min, float max);

public:
	virtual Value* greaterThanOrEqual(float f) const;
	virtual Value* equals(const Value* m) const;

public:
    virtual float reduceSum() const;
	virtual Value* reduceSumAlongColumns() const;
	virtual Value* reduceSumAlongRows() const;

public:
	virtual iterator       begin();
	virtual const_iterator begin() const;

	virtual iterator       end();
	virtual const_iterator end() const;

public:
	virtual       Matrix& front();
	virtual const Matrix& front() const;

	virtual       Matrix& back();
	virtual const Matrix& back() const;

public:
	virtual const Matrix& operator[](size_t position) const;
	virtual       Matrix& operator[](size_t position);

public:
	virtual void pop_back();
	virtual void push_back(const Matrix& m);

public:
    virtual size_t columns() const;
	virtual size_t rows()	const;

    virtual size_t columnsPerBlock() const;
	virtual size_t rowsPerBlock()    const;

public:
	virtual void resize(size_t blocks, size_t rowsPerBlock, size_t columnsPerBlock);
	virtual void resize(size_t blocks);

public:
	virtual MatrixVector& data();
	virtual const MatrixVector& data() const;

public:
	MatrixVector& rawData();

public:
	virtual Value* clone() const;

public:
	static bool isSupported();

private:
	void _performTransposeIfNecessary() const;
	void _performTransposeIfNecessary(const BlockSparseMatrixImplementation*) const;

private:
	bool _isTransposed;
};

}

}



