/*  \file   SoftmaxCostFunction.cpp
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the SoftmaxCostFunction class.
*/

// Lucius Includes
#include <lucius/network/interface/SoftmaxCostFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/SoftmaxOperations.h>

#include <lucius/util/interface/debug.h>

#include <string>

namespace lucius
{

namespace network
{

typedef matrix::Matrix Matrix;

SoftmaxCostFunction::~SoftmaxCostFunction()
{

}

Matrix SoftmaxCostFunction::computeCost(const Matrix& output, const Matrix& reference) const
{
    auto softmaxResult = softmax(output);

    auto result = apply(softmaxResult, matrix::Log());

    size_t samples = output.size()[output.size().size() - 2];

    return apply(apply(reference, result, matrix::Multiply()), matrix::Multiply(-1.0/samples));
}

Matrix SoftmaxCostFunction::computeDelta(const Matrix& output, const Matrix& reference) const
{
    return apply(softmax(output), reference, matrix::Subtract());
}

CostFunction* SoftmaxCostFunction::clone() const
{
    return new SoftmaxCostFunction;
}

std::string SoftmaxCostFunction::typeName() const
{
    return "SoftmaxCostFunction";
}

}

}



