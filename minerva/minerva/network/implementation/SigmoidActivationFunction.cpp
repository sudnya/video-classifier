/*	\file   SigmoidActivationFunction.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the SigmoidActivationFunction class.
*/

#include <minerva/neuralnetwork/interface/SigmoidLinearActivationFunction.h>

namespace minerva
{

namespace neuralnetwork
{

SigmoidActivationFunction::~RectifiedActivationFunction()
{

}

BlockSparseMatrix SigmoidActivationFunction::apply(const BlockSparseMatrix& activations) const
{
	return activations.sigmoid();
}

BlockSparseMatrix SigmoidActivationFunction::applyDerivative(const BlockSparseMatrix& deltas) const
{
	return deltas.sigmoidDerivative();
}

}

}


