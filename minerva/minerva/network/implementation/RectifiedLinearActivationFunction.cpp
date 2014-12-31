/*	\file   RectifiedLinearActivationFunction.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the RectifiedLinearActivationFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/RectifiedLinearActivationFunction.h>

#include <minerva/matrix/interface/BlockSparseMatrix.h>

namespace minerva
{

namespace network
{

typedef matrix::BlockSparseMatrix BlockSparseMatrix;

RectifiedLinearActivationFunction::~RectifiedLinearActivationFunction()
{

}

BlockSparseMatrix RectifiedLinearActivationFunction::apply(const BlockSparseMatrix& activations) const
{
	return activations.rectifiedLinear();
}

BlockSparseMatrix RectifiedLinearActivationFunction::applyDerivative(const BlockSparseMatrix& deltas) const
{
	return deltas.rectifiedLinearDerivative();
}

}

}



