/*	\file   RectifiedLinearActivationFunction.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the RectifiedLinearActivationFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/RectifiedLinearActivationFunction.h>

#include <minerva/matrix/interface/Matrix.h>

namespace minerva
{

namespace network
{

typedef matrix::Matrix Matrix;

RectifiedLinearActivationFunction::~RectifiedLinearActivationFunction()
{

}

Matrix RectifiedLinearActivationFunction::apply(const Matrix& activations) const
{
	return apply(activations, RectifiedLinear());
}

Matrix RectifiedLinearActivationFunction::applyDerivative(const Matrix& activations) const
{
	return apply(activations, RectifiedLinearDerivative());
}

ActivationFunction* RectifiedLinearActivationFunction::clone() const
{
	return new RectifiedLinearActivationFunction(*this);
}

}

}



