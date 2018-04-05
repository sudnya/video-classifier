/*  \file   CostFunction.cpp
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the CostFunction class.
*/

// Lucius Includes
#include <lucius/network/interface/CostFunction.h>

#include <lucius/network/interface/Bundle.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/GenericOperators.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/SoftmaxOperations.h>
#include <lucius/matrix/interface/RecurrentOperations.h>
#include <lucius/matrix/interface/DimensionTransformations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::MatrixVector MatrixVector;
typedef matrix::Dimension Dimension;
typedef std::vector<std::vector<size_t>> LabelVector;
typedef std::vector<size_t> IndexVector;

CostFunction::~CostFunction()
{

}

static void zeroEnds(Matrix& data, const LabelVector& labels)
{
    IndexVector lengths;

    for(auto& label : labels)
    {
        lengths.push_back(label.size());
    }

    recurrentZeroEnds(data, lengths);
}

void CostFunction::computeCost(Bundle& bundle) const
{
    computeCostImplementation(bundle);

    if(bundle.contains("outputActivationWeights"))
    {
        auto& costs = bundle["costs"].get<Matrix>();
        auto& labels = bundle["referenceLabels"].get<LabelVector>();

        zeroEnds(costs, labels);

        bundle["originalCosts"] = copy(costs);

        auto& weights = bundle["outputActivationWeights"].get<Matrix>();

        auto normalizedWeights = softmax(weights);

        auto flattenedWeights = flatten(normalizedWeights);

        Dimension broadcastDimensions = removeDimensions(range(costs.size()),
            {costs.size().size() - 2});

        broadcast(costs, costs, flattenedWeights, broadcastDimensions, matrix::Multiply());

        if(util::isLogEnabled("CostFunction::Detail"))
        {
            util::log("CostFunction::Detail") << " normalized weights : "
                << normalizedWeights.debugString();
            util::log("CostFunction::Detail") << " weighted costs : "
                << costs.debugString();
        }
    }
}

void CostFunction::computeDelta(Bundle& bundle) const
{
    computeDeltaImplementation(bundle);

    if(bundle.contains("outputActivationWeights"))
    {
        // compute output deltas
        auto& outputDeltasVector = bundle["outputDeltas"].get<MatrixVector>();
        auto& labels = bundle["referenceLabels"].get<LabelVector>();

        auto& outputActivationWeights = bundle["outputActivationWeights"].get<Matrix>();

        auto normalizedWeights = softmax(outputActivationWeights);

        auto flattenedWeights = flatten(normalizedWeights);

        auto& outputDeltas = outputDeltasVector.front();

        size_t beamSize      = outputActivationWeights.size()[0];
        size_t miniBatchSize = outputActivationWeights.size()[1];

        Dimension broadcastDimensions = removeDimensions(range(outputDeltas.size()),
            {outputDeltas.size().size() - 2});

        broadcast(outputDeltas, outputDeltas, flattenedWeights,
            broadcastDimensions, matrix::Multiply());

        zeroEnds(outputDeltas, labels);

        if(util::isLogEnabled("CostFunction::Detail"))
        {
            util::log("CostFunction::Detail") << " weighted output deltas : "
                << outputDeltas.debugString();
        }

        // compute weight deltas
        auto& costs = bundle["originalCosts"].get<Matrix>();

        if(util::isLogEnabled("CostFunction::Detail"))
        {
            util::log("CostFunction::Detail") << " original costs : "
                << costs.debugString();
        }

        auto outputDeltasScaledWithCosts = reshape(
            reduce(costs, broadcastDimensions, matrix::Add()), {beamSize, miniBatchSize});

        if(util::isLogEnabled("CostFunction::Detail"))
        {
            util::log("CostFunction::Detail") << " cost scaled output deltas : "
                << outputDeltasScaledWithCosts.debugString();
        }

        auto weightDeltas = softmaxGradient(normalizedWeights, outputDeltasScaledWithCosts);

        if(util::isLogEnabled("CostFunction::Detail"))
        {
            util::log("CostFunction::Detail") << " weight deltas : " << weightDeltas.debugString();
        }

        bundle["outputActivationWeightDeltas"] = weightDeltas;

    }
}

}

}


