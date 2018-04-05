/*    \file   RectifiedLinearActivationFunction.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the RectifiedLinearActivationFunction class.
*/

// Lucius Includes
#include <lucius/network/interface/RectifiedLinearActivationFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operator.h>
#include <lucius/matrix/interface/GenericOperators.h>

namespace lucius
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::Operator Operator;

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

Operator RectifiedLinearActivationFunction::getOperator() const
{
    return matrix::RectifiedLinear();
}

Operator RectifiedLinearActivationFunction::getDerivativeOperator() const
{
    return matrix::RectifiedLinearDerivative();
}

std::unique_ptr<ActivationFunction> RectifiedLinearActivationFunction::clone() const
{
    return std::make_unique<RectifiedLinearActivationFunction>(*this);
}

std::string RectifiedLinearActivationFunction::typeName() const
{
    return "RectifiedLinearActivationFunction";
}

}

}



