/*    \file   RectifiedLinearActivationFunction.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the RectifiedLinearActivationFunction class.
*/

// Lucius Includes
#include <lucius/network/interface/RectifiedLinearActivationFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>

namespace lucius
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::Operation Operation;

RectifiedLinearActivationFunction::~RectifiedLinearActivationFunction()
{

}

Matrix RectifiedLinearActivationFunction::apply(const Matrix& activations) const
{
    return matrix::apply(activations, matrix::RectifiedLinear());
}

Matrix RectifiedLinearActivationFunction::applyDerivative(const Matrix& activations) const
{
    return matrix::apply(activations, matrix::RectifiedLinearDerivative());
}

Operation RectifiedLinearActivationFunction::getOperation() const
{
    return matrix::RectifiedLinear();
}

Operation RectifiedLinearActivationFunction::getDerivativeOperation() const
{
    return matrix::RectifiedLinearDerivative();
}

ActivationFunction* RectifiedLinearActivationFunction::clone() const
{
    return new RectifiedLinearActivationFunction(*this);
}

std::string RectifiedLinearActivationFunction::typeName() const
{
    return "RectifiedLinearActivaitonFunction";
}

}

}



