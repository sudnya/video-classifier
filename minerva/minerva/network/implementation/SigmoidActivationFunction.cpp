/*    \file   SigmoidActivationFunction.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the SigmoidActivationFunction class.
*/

// Lucious Includes
#include <lucious/network/interface/SigmoidActivationFunction.h>

#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixOperations.h>
#include <lucious/matrix/interface/Operation.h>

namespace lucious
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::Operation Operation;

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

Operation SigmoidActivationFunction::getOperation() const
{
    return matrix::Sigmoid();
}

Operation SigmoidActivationFunction::getDerivativeOperation() const
{
    return matrix::SigmoidDerivative();
}

ActivationFunction* SigmoidActivationFunction::clone() const
{
    return new SigmoidActivationFunction(*this);
}

}

}


