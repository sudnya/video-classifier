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
typedef matrix::IndexVector IndexVector;
typedef matrix::LabelVector LabelVector;

CTCCostFunction::~CTCCostFunction()
{

}

void CTCCostFunction::computeCostImplementation(Bundle& bundle) const
{
    auto& output    = bundle["outputActivations"].get<MatrixVector>().front();
    auto& labels    = bundle["referenceLabels"].get<LabelVector>();
    auto& timesteps = bundle["inputTimesteps"].get<IndexVector>();

    size_t miniBatchSize = output.size()[output.size().size() - 2];

    Matrix cost({miniBatchSize}, output.precision());
    Matrix fakeGradients;

    matrix::computeCtc(cost, fakeGradients, output, labels, timesteps);

    bundle["costs"] = apply(cost, matrix::Divide(miniBatchSize));
}

void CTCCostFunction::computeDeltaImplementation(Bundle& bundle) const
{
    auto& output    = bundle["outputActivations"].get<MatrixVector>().front();
    auto& labels    = bundle["referenceLabels"].get<LabelVector>();
    auto& timesteps = bundle["inputTimesteps"].get<IndexVector>();

    size_t miniBatchSize = output.size()[output.size().size() - 2];

    Matrix cost({miniBatchSize}, output.precision());
    Matrix gradients = zeros(output.size(), output.precision());

    matrix::computeCtc(cost, gradients, output, labels, timesteps);

    bundle["outputDeltas"] = MatrixVector({apply(gradients, matrix::Divide(miniBatchSize))});
}

std::unique_ptr<CostFunction> CTCCostFunction::clone() const
{
    return std::make_unique<CTCCostFunction>();
}

std::string CTCCostFunction::typeName() const
{
    return "CTCCostFunction";
}

}

}




