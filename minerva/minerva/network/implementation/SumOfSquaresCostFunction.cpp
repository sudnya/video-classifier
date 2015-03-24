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
	auto difference = apply(output, reference, Subtract());

	size_t samples = output.size()[0];

	return apply(difference, SquareAndScale(0.5*samples));
}

BlockSparseMatrix SumOfSquaresCostFunction::computeDelta(const BlockSparseMatrix& output, const BlockSparseMatrix& reference) const
{
	return apply(output, reference, Subtract());
}

CostFunction* SumOfSquaresCostFunction::clone() const
{
	return new SumOfSquaresCostFunction;
}

}

}

