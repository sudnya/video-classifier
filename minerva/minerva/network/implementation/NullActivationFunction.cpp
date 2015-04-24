/*	\file   NullActivationFunction.cpp
	\date   April 23, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NullActivationFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/NullActivationFunction.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixOperations.h>

namespace minerva
{

namespace network
{

typedef matrix::Matrix Matrix;

NullActivationFunction::~NullActivationFunction()
{

}

Matrix NullActivationFunction::apply(const Matrix& activations) const
{
	return activations;
}

Matrix NullActivationFunction::applyDerivative(const Matrix& activations) const
{
	return ones(activations.size(), activations.precision());
}

ActivationFunction* NullActivationFunction::clone() const
{
	return new NullActivationFunction(*this);
}

}

}




