/*! \file:   SparseMatrixFormat.cpp
	\author: Gregory Diamos <gregory.diamos@gatech.edu>
	\date:   Tuesday January 21, 2014
	\brief   The source file for the SparseMatrixFormat class.
*/

// Minerva Includes
#include <minerva/matrix/interface/SparseMatrixFormat.h>

#include <minerva/matrix/interface/BlockSparseMatrix.h>
#include <minerva/matrix/interface/Matrix.h>

namespace minerva
{

namespace matrix
{

SparseMatrixFormat::SparseMatrixFormat(size_t b, size_t r, size_t c, bool s)
: blocks(b), rowsPerBlock(r), columnsPerBlock(c), isRowSparse(s)
{

}

SparseMatrixFormat::SparseMatrixFormat(const BlockSparseMatrix& matrix)
: blocks(matrix.blocks()), rowsPerBlock(matrix.rowsPerBlock()),
  columnsPerBlock(matrix.columnsPerBlock()), isRowSparse(matrix.isRowSparse())
{

}

SparseMatrixFormat::SparseMatrixFormat(const Matrix& matrix)
: blocks(1), rowsPerBlock(matrix.rows()), columnsPerBlock(matrix.columns()), isRowSparse(true)
{

}

}

}



