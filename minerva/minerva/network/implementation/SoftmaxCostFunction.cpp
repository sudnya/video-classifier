/*    \file   SoftmaxCostFunction.cpp
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the SoftmaxCostFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/SoftmaxCostFunction.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/util/interface/debug.h>

#include <string>

namespace minerva
{

namespace network
{

typedef matrix::Matrix Matrix;

SoftmaxCostFunction::~SoftmaxCostFunction()
{

}

static Matrix softmax(const Matrix& output)
{
    auto normalizedOutput = broadcast(output, reduce(output, {0}, matrix::Maximum()), {0}, matrix::Subtract());

    auto expOutput = apply(normalizedOutput, matrix::Exp());

    auto sums = reduce(expOutput, {0}, matrix::Add());

    return broadcast(expOutput, sums, {0}, matrix::Divide());
}

Matrix SoftmaxCostFunction::computeCost(const Matrix& output, const Matrix& reference) const
{
    auto softmaxResult = softmax(output);

    auto result = apply(softmaxResult, matrix::Log());

    size_t samples = output.size()[1];

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

}

}



