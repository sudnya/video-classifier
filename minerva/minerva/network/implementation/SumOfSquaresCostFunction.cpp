/*	\file   SumOfSquaresCostFunction.cpp
	\date   November 19, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the SumOfSquaresCostFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/SumOfSquaresCostFunction.h>

#include <minerva/matrix/interface/BlockSparseMatrix.h>

namespace minerva
{

namespace network
{

typedef matrix::BlockSparseMatrix BlockSparseMatrix;

SumOfSquaresCostFunction::~SumOfSquaresCostFunction()
{

}

BlockSparseMatrix SumOfSquaresCostFunction::computeCost(const BlockSparseMatrix& output, const BlockSparseMatrix& reference) const
{
	auto difference = output.subtract(reference);

	size_t samples = output.rows();

	return difference.elementMultiply(difference).multiply(1.0f / (samples * 2.0f));
}

BlockSparseMatrix SumOfSquaresCostFunction::computeDelta(const BlockSparseMatrix& output, const BlockSparseMatrix& reference) const
{
	return output.subtract(reference);
}

CostFunction* SumOfSquaresCostFunction::clone() const
{
	return new SumOfSquaresCostFunction;
}

}

}

