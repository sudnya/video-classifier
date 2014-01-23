/*! \file:   SparseMatrixFormat.h
	\author: Gregory Diamos <gregory.diamos@gatech.edu>
	\date:   Tuesday January 21, 2014
	\brief   The header file for the SparseMatrixFormat class.
*/

#pragma once

// Standard Library Includes
#include <cstddef>
#include <vector>

// Forward Declarations
namespace minerva { namespace matrix    { class Matrix;             } }
namespace minerva { namespace matrix    { class BlockSparseMatrix;  } }

namespace minerva
{

namespace optimizer
{

/* \brief An abstract representation of a sparse matrix */
class SparseMatrixFormat
{
public:
	explicit SparseMatrixFormat(size_t blocks = 0, size_t rowsPerBlock = 0,
		size_t columnsPerBlock = 0, bool isRowSparse = true);
	SparseMatrixFormat(const matrix::BlockSparseMatrix& );
	SparseMatrixFormat(const matrix::Matrix& );
	
public:
	size_t blocks;
	size_t rowsPerBlock;
	size_t columnsPerBlock;
	bool   isRowSparse;
};

typedef std::vector<SparseMatrixFormat> SparseMatrixVectorFormat;

}

}


