/*    \file   NullActivationFunction.cpp
    \date   April 23, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the NullActivationFunction class.
*/

// Lucius Includes
#include <lucius/network/interface/NullActivationFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/Operation.h>

namespace lucius
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::Operation Operation;

NullActivationFunction::~NullActivationFunction()
{

}

Matrix NullActivationFunction::apply(const Matrix& activations) const
{
    return copy(activations);
}

Matrix NullActivationFunction::applyDerivative(const Matrix& activations) const
{
    return ones(activations.size(), activations.precision());
}

Operation NullActivationFunction::getOperation() const
{
    return matrix::Nop();
}

Operation NullActivationFunction::getDerivativeOperation() const
{
    return matrix::NopDerivative();
}

ActivationFunction* NullActivationFunction::clone() const
{
    return new NullActivationFunction(*this);
}

std::string NullActivationFunction::typeName() const
{
    return "NullActivationFunction";
}

}

}




