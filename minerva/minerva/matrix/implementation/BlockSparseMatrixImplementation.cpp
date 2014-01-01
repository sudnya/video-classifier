/*! \file   BlockSparseMatrixImplementation.cpp
	\author Gregory Diamos
	\date   Sunday December 29, 2013
	\brief  The source file for the BlockSparseMatrixImplementatio class.
*/

// Minerva Includes
#include <minerva/matrix/interface/BlockSparseMatrixImplementation.h>
#include <minerva/matrix/interface/NaiveBlockSparseMatrix.h>
#include <minerva/matrix/interface/CudaBlockSparseMatrix.h>
#include <minerva/matrix/interface/Matrix.h>

namespace minerva
{

namespace matrix
{

typedef BlockSparseMatrixImplementation::Value Value;
typedef BlockSparseMatrixImplementation::MatrixVector MatrixVector;

BlockSparseMatrixImplementation::BlockSparseMatrixImplementation(size_t blocks, size_t rows,
	size_t columns, bool rowSparse)
: _isRowSparse(rowSparse)
{
	
}

BlockSparseMatrixImplementation::~BlockSparseMatrixImplementation()
{

}

BlockSparseMatrixImplementation::iterator BlockSparseMatrixImplementation::begin()
{
	return _matrices.begin();
}

BlockSparseMatrixImplementation::const_iterator BlockSparseMatrixImplementation::begin() const
{
	return _matrices.begin();
}

BlockSparseMatrixImplementation::iterator BlockSparseMatrixImplementation::end()
{
	return _matrices.end();
}

BlockSparseMatrixImplementation::const_iterator BlockSparseMatrixImplementation::end() const
{
	return _matrices.end();
}

Matrix& BlockSparseMatrixImplementation::front()
{
	return _matrices.front();
}

const Matrix& BlockSparseMatrixImplementation::front() const
{
	return _matrices.front();
}

Matrix& BlockSparseMatrixImplementation::back()
{
	return _matrices.back();
}

const Matrix& BlockSparseMatrixImplementation::back() const
{
	return _matrices.back();
}

const Matrix& BlockSparseMatrixImplementation::operator[](size_t position) const
{
	return _matrices[position];
}

Matrix& BlockSparseMatrixImplementation::operator[](size_t position)
{
	return _matrices[position];
}

void BlockSparseMatrixImplementation::pop_back()
{
	return _matrices.pop_back();
}

void BlockSparseMatrixImplementation::push_back(const Matrix& m)
{
	return _matrices.push_back(m);
}

size_t BlockSparseMatrixImplementation::size() const
{
	size_t s = 0;

	for(auto& m : *this)
	{
		s += m.size();
	}

	return s;
}

size_t BlockSparseMatrixImplementation::blocks() const
{
	if(empty()) return 0;
	
	if(isRowSparse()) return front().rows();

	return front().columns();
}

bool BlockSparseMatrixImplementation::empty() const
{
	return _matrices.empty();
}

size_t BlockSparseMatrixImplementation::columns() const
{
	if(isColumnSparse())
	{
		size_t c = 0;

		for(auto& matrix : *this)
		{
			c += matrix.columns();
		}

		return c;
	}
	
	return front().columns();
}

size_t BlockSparseMatrixImplementation::rows() const
{
	if(isRowSparse())
	{
		size_t r = 0;

		for(auto& matrix : *this)
		{
			r += matrix.rows();
		}

		return r;
	}

	return front().rows();
}
    
size_t BlockSparseMatrixImplementation::columnsPerBlock() const
{
	if(empty()) return 0;
	
	return front().columns();
}

size_t BlockSparseMatrixImplementation::rowsPerBlock() const
{
	if(empty()) return 0;
	
	return front().rows();
}

MatrixVector& BlockSparseMatrixImplementation::data()
{
	return _matrices;
}

const MatrixVector& BlockSparseMatrixImplementation::data() const
{
	return _matrices;
}
	
void BlockSparseMatrixImplementation::resize(size_t blocks, size_t rowsPerBlock, size_t columnsPerBlock)
{
	_matrices.resize(blocks);
	
	for(auto& matrix : *this)
	{
		matrix.resize(rowsPerBlock, columnsPerBlock);
	}
}

void BlockSparseMatrixImplementation::resize(size_t blocks)
{
	_matrices.resize(blocks);
}

bool& BlockSparseMatrixImplementation::isRowSparse()
{
	return _isRowSparse;
}

bool BlockSparseMatrixImplementation::isRowSparse() const
{
	return _isRowSparse;
}

bool BlockSparseMatrixImplementation::isColumnSparse() const
{
	return not isRowSparse();
}

Value* BlockSparseMatrixImplementation::createBestImplementation(size_t blocks, size_t rows,
	size_t columns, bool isRowSparse)
{
	Value* matrix = nullptr;
	
	if(matrix == nullptr && CudaBlockSparseMatrix::isSupported())
	{
		matrix = new CudaBlockSparseMatrix(blocks, rows, columns, isRowSparse);
	}
	
	if(matrix == nullptr)
	{	
		matrix = new NaiveBlockSparseMatrix(blocks, rows, columns, isRowSparse);
	}
	
	return matrix;
}

}

}


