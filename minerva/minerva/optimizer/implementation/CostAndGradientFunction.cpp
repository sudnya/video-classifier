/*! \file:   CostAndGradientFunction.cpp
	\author: Gregory Diamos <gregory.diamos@gatech.edu>
	\date:   Tuesday January 21, 2014
	\brief   The source file for the CostAndGradientFunction class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/CostAndGradientFunction.h>
#include <minerva/optimizer/interface/SparseMatrixFormat.h>

#include <minerva/matrix/interface/BlockSparseMatrix.h>
#include <minerva/matrix/interface/BlockSparseMatrixVector.h>

namespace minerva
{

namespace optimizer
{

typedef matrix::Matrix                  Matrix;
typedef matrix::BlockSparseMatrix       BlockSparseMatrix;
typedef matrix::BlockSparseMatrixVector BlockSparseMatrixVector;

CostAndGradientFunction::CostAndGradientFunction(const SparseMatrixVectorFormat& f)
: format(f)
{

}

static SparseMatrixVectorFormat convertToFormat(const BlockSparseMatrixVector& vector)
{
	SparseMatrixVectorFormat format;
	
	for(auto& matrix : vector)
	{
		format.push_back(SparseMatrixFormat(matrix));
	}
	
	return format;
}

static SparseMatrixVectorFormat convertToFormat(const Matrix& matrix)
{
	SparseMatrixVectorFormat format;
	
	format.push_back(SparseMatrixFormat(matrix));
	
	return format;
}

CostAndGradientFunction::CostAndGradientFunction(const BlockSparseMatrixVector& vector)
: format(convertToFormat(vector))
{

}

CostAndGradientFunction::CostAndGradientFunction(const Matrix& matrix)
: format(convertToFormat(matrix))
{

}

CostAndGradientFunction::~CostAndGradientFunction()
{

}

BlockSparseMatrixVector CostAndGradientFunction::getUninitializedDataStructure() const
{
	BlockSparseMatrixVector vector;
	
	vector.reserve(format.size());
	
	for(auto& sparseMatrixFormat : format)
	{
		vector.push_back(BlockSparseMatrix(sparseMatrixFormat.blocks,
			sparseMatrixFormat.rowsPerBlock,
			sparseMatrixFormat.columnsPerBlock,
			sparseMatrixFormat.isRowSparse));
	}
	
	return vector;
}

}

}


