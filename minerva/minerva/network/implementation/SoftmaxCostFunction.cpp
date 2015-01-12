/*	\file   SoftmaxCostFunction.cpp
	\date   November 19, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the SoftmaxCostFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/SoftmaxCostFunction.h>

#include <minerva/matrix/interface/BlockSparseMatrix.h>

#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace network
{

typedef matrix::BlockSparseMatrix BlockSparseMatrix;

SoftmaxCostFunction::~SoftmaxCostFunction()
{

}

BlockSparseMatrix SoftmaxCostFunction::computeCost(const BlockSparseMatrix& output, const BlockSparseMatrix& reference) const
{
	assertM(false, "Not implemented.");
}

BlockSparseMatrix SoftmaxCostFunction::computeDelta(const BlockSparseMatrix& output, const BlockSparseMatrix& reference) const
{
	assertM(false, "Not implemented.");
}

CostFunction* SoftmaxCostFunction::clone() const
{
	return new SoftmaxCostFunction;
}

}

}



