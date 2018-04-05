/*  \file   SoftmaxCostFunction.cpp
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the SoftmaxCostFunction class.
*/

// Lucius Includes
#include <lucius/network/interface/SoftmaxCostFunction.h>
#include <lucius/network/interface/Bundle.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/GenericOperators.h>
#include <lucius/matrix/interface/SoftmaxOperations.h>

#include <lucius/util/interface/debug.h>

#include <string>

namespace lucius
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::MatrixVector MatrixVector;

SoftmaxCostFunction::~SoftmaxCostFunction()
{

}

void SoftmaxCostFunction::computeCostImplementation(Bundle& bundle) const
{
    auto& output    = bundle["outputActivations"].get<MatrixVector>().front();
    auto& reference = bundle["referenceActivations"].get<MatrixVector>().front();

    auto softmaxResult = logsoftmax(output);

    auto result = softmaxResult;

    size_t samples = output.size()[output.size().size() - 2];

    bundle["costs"] = apply(apply(Matrix(reference), result, matrix::Multiply()),
        matrix::Multiply(-1.0/samples));
}

void SoftmaxCostFunction::computeDeltaImplementation(Bundle& bundle) const
{
    auto& output    = bundle["outputActivations"].get<MatrixVector>().front();
    auto& reference = bundle["referenceActivations"].get<MatrixVector>().front();

    bundle["outputDeltas"] = MatrixVector({apply(softmax(output), reference, matrix::Subtract())});
}

std::unique_ptr<CostFunction> SoftmaxCostFunction::clone() const
{
    return std::make_unique<SoftmaxCostFunction>();
}

std::string SoftmaxCostFunction::typeName() const
{
    return "SoftmaxCostFunction";
}

}

}



