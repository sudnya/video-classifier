/*    \file   RectifiedLinearActivationFunction.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the RectifiedLinearActivationFunction class.
*/

// Lucious Includes
#include <lucious/network/interface/RectifiedLinearActivationFunction.h>

#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixOperations.h>
#include <lucious/matrix/interface/Operation.h>

namespace lucious
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



