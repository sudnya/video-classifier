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
namespace minerva { namespace matrix { class Matrix;                          } }
namespace minerva { namespace matrix { class BlockSparseMatrixImplementation; } }
namespace minerva { namespace matrix { class SparseMatrixFormat;              } }

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
	
	typedef       float& FloatReference;
	typedef const float& ConstFloatReference;
	typedef       float* FloatPointer;
	typedef const float* ConstFloatPointer;

public:
	explicit BlockSparseMatrix(size_t blocks = 0, size_t rows = 0,
		size_t columns = 0, bool rowSparse = true);
	explicit BlockSparseMatrix(bool rowSparse);

	BlockSparseMatrix(const BlockSparseMatrix&);
	BlockSparseMatrix(BlockSparseMatrix&&);

public:
	~BlockSparseMatrix();

public:
	BlockSparseMatrix& operator=(const BlockSparseMatrix&);
	BlockSparseMatrix& operator=(BlockSparseMatrix&&);

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
	     
		 FloatReference operator()(size_t row, size_t column);
	ConstFloatReference operator()(size_t row, size_t column) const;

public:
	void pop_back();
	void push_back(const Matrix& m);

public:
	size_t size()   const;
	size_t blocks() const;
	bool   empty()  const;

    size_t columns() const;
	size_t rows()	const;

	size_t rowsPerBlock()    const;
	size_t columnsPerBlock() const;

	bool isRowSparse()    const;
	bool isColumnSparse() const;

	size_t getBlockingFactor() const;
	size_t blockSize() const;

public:
	void resize(size_t blocks, size_t rowsPerBlock, size_t columnsPerBlock);
	void resize(size_t blocks);
	void setColumnSparse();
	void setRowSparse();

public: 
	BlockSparseMatrix multiply(const BlockSparseMatrix& m) const;
	BlockSparseMatrix convolutionalMultiply(const BlockSparseMatrix& m, size_t step) const;

	BlockSparseMatrix computeConvolutionalGradient(const BlockSparseMatrix& activation,
		const SparseMatrixFormat& weightFormat, size_t step) const;
	BlockSparseMatrix computeConvolutionalBiasGradient(const SparseMatrixFormat& activationFormat,
		const SparseMatrixFormat& weightFormat, size_t step) const;
	BlockSparseMatrix computeConvolutionalDeltas(const BlockSparseMatrix& weights,
		const SparseMatrixFormat& deltasFormat, size_t step) const;

	BlockSparseMatrix multiply(float f) const;
	BlockSparseMatrix elementMultiply(const BlockSparseMatrix& m) const;

	BlockSparseMatrix add(const BlockSparseMatrix& m) const;
	BlockSparseMatrix convolutionalAddBroadcastRow(const BlockSparseMatrix& m) const;
	BlockSparseMatrix addBroadcastRow(const BlockSparseMatrix& m) const;
	BlockSparseMatrix add(float f) const;

	BlockSparseMatrix subtract(const BlockSparseMatrix& m) const;
	BlockSparseMatrix subtract(float f) const;

	BlockSparseMatrix log() const;
	BlockSparseMatrix negate() const;

	BlockSparseMatrix sigmoid() const;
	BlockSparseMatrix sigmoidDerivative() const;
	
	BlockSparseMatrix rectifiedLinear() const;
	BlockSparseMatrix rectifiedLinearDerivative() const;
	
	BlockSparseMatrix klDivergence(float sparsity) const;
	BlockSparseMatrix klDivergenceDerivative(float sparsity) const;

public:
	BlockSparseMatrix transpose() const;

public:
	void negateSelf();
	void logSelf();
    void sigmoidSelf();
    void sigmoidDerivativeSelf();
    void rectifiedLinearSelf();
    void rectifiedLinearDerivativeSelf();
	void minSelf(float f);
	void maxSelf(float f);
	void assignSelf(float f);

	void transposeSelf();

	void assignUniformRandomValues(std::default_random_engine& engine,
		float min = 0.0f, float max = 1.0f);

public:
	Matrix slice(size_t startRow, size_t startColumn,
		size_t rows, size_t columns) const;
	void assign(size_t startRow, size_t startColumn, const Matrix&);

public:
	BlockSparseMatrix greaterThanOrEqual(float f) const;
	BlockSparseMatrix equals(const BlockSparseMatrix& m) const;

public:
	Matrix toMatrix() const;

public:
    float reduceSum() const;
	BlockSparseMatrix reduceSumAlongColumns() const;
	BlockSparseMatrix reduceSumAlongRows() const;
	BlockSparseMatrix reduceTileSumAlongRows(size_t rowsPerTile, size_t blocks) const;

public:
	bool operator==(const BlockSparseMatrix& m) const;
	bool operator!=(const BlockSparseMatrix& m) const;

public:
	std::string toString()    const;
	std::string debugString() const;
	std::string shapeString() const;

private:
	explicit BlockSparseMatrix(BlockSparseMatrixImplementation*);

private:
	BlockSparseMatrixImplementation* _implementation;

};

}

}


