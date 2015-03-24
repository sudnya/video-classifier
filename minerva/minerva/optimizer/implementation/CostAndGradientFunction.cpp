/*! \file:   CostAndGradientFunction.cpp
	\author: Gregory Diamos <gregory.diamos@gatech.edu>
	\date:   Tuesday January 21, 2014
	\brief   The source file for the CostAndGradientFunction class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/CostAndGradientFunction.h>

#include <minerva/matrix/interface/SparseMatrixFormat.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixVector.h>

namespace minerva
{

namespace optimizer
{

typedef matrix::Matrix                   Matrix;
typedef matrix::MatrixVector             MatrixVector;
typedef matrix::SparseMatrixFormat       SparseMatrixFormat;
typedef matrix::SparseMatrixVectorFormat SparseMatrixVectorFormat;

CostAndGradientFunction::CostAndGradientFunction(const SparseMatrixVectorFormat& f)
: format(f)
{

}

static SparseMatrixVectorFormat convertToFormat(const MatrixVector& vector)
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

CostAndGradientFunction::CostAndGradientFunction(const MatrixVector& vector)
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

MatrixVector CostAndGradientFunction::getUninitializedDataStructure() const
{
	MatrixVector vector;
	
	vector.reserve(format.size());
	
	for(auto& sparseMatrixFormat : format)
	{
		vector.push_back(Matrix(sparseMatrixFormat.blocks,
			sparseMatrixFormat.rowsPerBlock,
			sparseMatrixFormat.columnsPerBlock,
			sparseMatrixFormat.isRowSparse));
	}
	
	return vector;
}

}

}


