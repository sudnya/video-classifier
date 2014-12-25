/*	\file   RectifiedLinearActivationFunction.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the RectifiedLinearActivationFunction class.
*/

#include <minerva/neuralnetwork/interface/RectifiedLinearActivationFunction.h>

namespace minerva
{

namespace neuralnetwork
{

RectifiedLinearActivationFunction::~RectifiedActivationFunction()
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



