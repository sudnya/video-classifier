/*! \file   BlockSparseMatrixImplementation.h
	\author Gregory Diamos
	\date   Sunday December 29, 2013
	\brief  The header file for the BlockSparseMatrixImplementatio class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <random>

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }

namespace minerva
{

namespace matrix
{

/*! \brief A an interface for a block sparse matrix */
class BlockSparseMatrixImplementation
{
public:
	typedef std::vector<Matrix> MatrixVector;
	typedef std::vector<float> FloatVector;

	typedef MatrixVector::iterator       iterator;
	typedef MatrixVector::const_iterator const_iterator;
	
	typedef BlockSparseMatrixImplementation Value;

public:
	explicit BlockSparseMatrixImplementation(size_t blocks, size_t rows,
		size_t columns, bool rowSparse);
	virtual ~BlockSparseMatrixImplementation();

public: 
	virtual Value* multiply(const Value* m) const = 0;
	virtual Value* convolutionalMultiply(const Value* m, size_t step) const = 0;
	virtual Value* reverseConvolutionalMultiply(const Value* m) const = 0;
	virtual Value* multiply(float f) const = 0;
	virtual Value* elementMultiply(const Value* m) const = 0;

	virtual Value* add(const Value* m) const = 0;
	virtual Value* addBroadcastRow(const Value* m) const = 0;
	virtual Value* convolutionalAddBroadcastRow(const Value* m) const = 0;
	virtual Value* add(float f) const = 0;

	virtual Value* subtract(const Value* m) const = 0;
	virtual Value* subtract(float f) const = 0;

	virtual Value* log() const = 0;
	virtual Value* negate() const = 0;
	virtual Value* sigmoid() const = 0;
	virtual Value* sigmoidDerivative() const = 0;
	
	virtual Value* klDivergence(float sparsity) const = 0;
	virtual Value* klDivergenceDerivative(float sparsity) const = 0;

public:
	virtual Value* transpose() const = 0;

public:
	virtual void negateSelf() = 0;
	virtual void logSelf() = 0;
    virtual void sigmoidSelf() = 0;
    virtual void sigmoidDerivativeSelf() = 0;
	virtual void minSelf(float v) = 0;
	virtual void maxSelf(float v) = 0;
	virtual void assignSelf(float v) = 0;
	
	virtual void transposeSelf() = 0;

	virtual void assignUniformRandomValues(std::default_random_engine& engine,
		float min, float max) = 0;

public:
	virtual Value* greaterThanOrEqual(float f) const = 0;
	virtual Value* equals(const Value* m) const = 0;

public:
    virtual float reduceSum() const = 0;
	virtual Value* reduceSumAlongColumns() const = 0;
	virtual Value* reduceSumAlongRows() const = 0;
	virtual Value* reduceTileSumAlongRows(size_t tilesPerRow, size_t blocks) const = 0;

public:
	virtual Value* clone() const = 0;

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
	size_t size()   const;
	size_t blocks() const;
	bool   empty()  const;

	size_t getBlockingFactor() const;

public:
    virtual size_t columns() const;
	virtual size_t rows()	const;

    virtual size_t columnsPerBlock() const;
	virtual size_t rowsPerBlock()    const;

public:
	virtual void resize(size_t blocks, size_t rowsPerBlock, size_t columnsPerBlock);
	virtual void resize(size_t blocks);

public:
	Matrix toMatrix() const;

public:
	virtual MatrixVector& data();
	virtual const MatrixVector& data() const;

public:
	std::string toString()    const;
	std::string debugString() const;
	std::string shapeString() const;

public:
	bool& isRowSparse();
	bool isRowSparse() const;
	bool isColumnSparse() const;

public:
	static Value* createBestImplementation(size_t blocks, size_t rows,
		size_t columns, bool isRowSparse);

protected:
	MatrixVector _matrices;
	bool         _isRowSparse;
	

};

}

}






