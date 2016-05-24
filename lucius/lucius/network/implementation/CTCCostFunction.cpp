/*  \file   CTCCostFunction.cpp
    \date   Feb 24th, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The source file for the CTCCostFunction class.
*/

// Lucius Includes
#include <lucius/network/interface/CTCCostFunction.h>

#include <lucius/network/interface/Bundle.h>

#include <lucius/matrix/interface/CTCOperations.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
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

CTCCostFunction::~CTCCostFunction()
{

}

void CTCCostFunction::computeCost(Bundle& bundle) const
{
    auto& output    = bundle["outputActivations"].get<matrix::MatrixVector>().front();
    auto& reference = bundle["referenceActivations"].get<matrix::MatrixVector>().front();

    size_t miniBatchSize = output.size()[output.size().size() - 2];

    Matrix cost({miniBatchSize}, output.precision());
    Matrix fakeGradients;

    matrix::computeCtc(cost, fakeGradients, output, reference);

    bundle["costs"] = apply(cost, matrix::Divide(miniBatchSize));
}

void CTCCostFunction::computeDelta(Bundle& bundle) const
{
    auto& output    = bundle["outputActivations"].get<matrix::MatrixVector>().front();
    auto& reference = bundle["referenceActivations"].get<matrix::MatrixVector>().front();

    size_t miniBatchSize = output.size()[output.size().size() - 2];

    Matrix cost({miniBatchSize}, output.precision());
    Matrix gradients = zeros(output.size(), output.precision());

    matrix::computeCtc(cost, gradients, output, reference);

    bundle["outputDeltas"] = matrix::MatrixVector(
        {apply(gradients, matrix::Divide(miniBatchSize))});
}

CostFunction* CTCCostFunction::clone() const
{
    return new CTCCostFunction;
}

std::string CTCCostFunction::typeName() const
{
    return "CTCCostFunction";
}

}

}




