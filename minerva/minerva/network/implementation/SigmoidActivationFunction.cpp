/*	\file   SigmoidActivationFunction.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the SigmoidActivationFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/SigmoidActivationFunction.h>

#include <minerva/matrix/interface/BlockSparseMatrix.h>

namespace minerva
{

namespace network
{

typedef matrix::BlockSparseMatrix BlockSparseMatrix;

SigmoidActivationFunction::~SigmoidActivationFunction()
{

}

BlockSparseMatrix SigmoidActivationFunction::apply(const BlockSparseMatrix& activations) const
{
	return activations.sigmoid();
}

BlockSparseMatrix SigmoidActivationFunction::applyDerivative(const BlockSparseMatrix& activations) const
{
	return activations.sigmoidDerivative();
}

ActivationFunction* SigmoidActivationFunction::clone() const
{
	return new SigmoidActivationFunction(*this);
}

}

}


