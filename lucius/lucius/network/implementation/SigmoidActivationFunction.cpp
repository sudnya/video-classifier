/*    \file   SigmoidActivationFunction.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the SigmoidActivationFunction class.
*/

// Lucius Includes
#include <lucius/network/interface/SigmoidActivationFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/GenericOperators.h>
#include <lucius/matrix/interface/Operator.h>

namespace lucius
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::Operator Operator;

SigmoidActivationFunction::~SigmoidActivationFunction()
{

}

Matrix SigmoidActivationFunction::apply(const Matrix& activations) const
{
    return matrix::apply(activations, matrix::Sigmoid());
}

Matrix SigmoidActivationFunction::applyDerivative(const Matrix& activations) const
{
    return matrix::apply(activations, matrix::SigmoidDerivative());
}

Operator SigmoidActivationFunction::getOperator() const
{
    return matrix::Sigmoid();
}

Operator SigmoidActivationFunction::getDerivativeOperator() const
{
    return matrix::SigmoidDerivative();
}

std::unique_ptr<ActivationFunction> SigmoidActivationFunction::clone() const
{
    return std::make_unique<SigmoidActivationFunction>(*this);
}

std::string SigmoidActivationFunction::typeName() const
{
    return "SigmoidActivationFunction";
}

}

}


