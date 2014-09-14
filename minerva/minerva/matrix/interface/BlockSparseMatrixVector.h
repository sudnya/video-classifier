
#pragma once

#include <minerva/matrix/interface/BlockSparseMatrix.h>

namespace minerva
{

namespace matrix
{

class BlockSparseMatrixVector
{
private:
	typedef BlockSparseMatrix value_type;
	typedef value_type& reference_type;
	typedef const value_type& const_reference_type;

public:
	typedef std::vector<value_type> Vector;

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
	reference_type operator[](size_t i);
	const_reference_type operator[](size_t i) const;

public:
	BlockSparseMatrixVector negate() const;

public:
	BlockSparseMatrixVector subtract(const BlockSparseMatrixVector& ) const;
	BlockSparseMatrixVector add(const BlockSparseMatrixVector& ) const;
	BlockSparseMatrixVector elementMultiply(const BlockSparseMatrixVector& ) const;

public:
	BlockSparseMatrixVector add(float ) const;
	BlockSparseMatrixVector multiply(float) const;

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



