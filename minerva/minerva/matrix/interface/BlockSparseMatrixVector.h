
#pragma once

#include <minerva/matrix/interface/BlockSparseMatrix.h>

namespace minerva
{

namespace matrix
{

class BlockSparseMatrixVector
{
private:
	typedef std::vector<BlockSparseMatrix> Vector;

public:
	typedef Vector::iterator       iterator;
	typedef Vector::const_iterator const_iterator;

	typedef Vector::reverse_iterator       reverse_iterator;
	typedef Vector::const_reverse_iterator const_reverse_iterator;

public:
	BlockSparseMatrixVector();

public:
	BlockSparseMatrixVector(const BlockSparseMatrixVector&);
	BlockSparseMatrixVector(const BlockSparseMatrixVector&&);

public:
	BlockSparseMatrixVector& operator=(const BlockSparseMatrixVector&  );
	BlockSparseMatrixVector& operator=(const BlockSparseMatrixVector&& );

public:
	BlockSparseMatrixVector negate() const;

public:
	BlockSparseMatrixVector subtract(const BlockSparseMatrixVector& ) const;
	BlockSparseMatrixVector elementMultiply(const BlockSparseMatrixVector& ) const;

public:
	BlockSparseMatrixVector add(float ) const;

public:
	void addSelf(const BlockSparseMatrixVector& );
	void multiplySelf(float );

public:
	float dotProduct(const BlockSparseMatrixVector& ) const;

public:
	float reduceSum() const;

public:
	bool empty() const;

public:
	size_t size() const;

public:
	void reserve(size_t size);
	void resize(size_t size);

public:
	iterator       begin();
	const_iterator begin() const;

	iterator       end();
	const_iterator end() const;

public:
	reverse_iterator       rbegin();
	const_reverse_iterator rbegin() const;
	
	reverse_iterator       rend();
	const_reverse_iterator rend() const;

public:
	void push_back(const BlockSparseMatrix&  );
	void push_back(const BlockSparseMatrix&& );

private:
	Vector _matrix;

};

}

}



