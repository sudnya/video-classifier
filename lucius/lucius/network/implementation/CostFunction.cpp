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
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/SoftmaxOperations.h>
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

CostFunction::~CostFunction()
{

}

void CostFunction::computeCost(Bundle& bundle) const
{
    computeCostImplementation(bundle);

    if(bundle.contains("outputActivationWeights"))
    {
        auto& costs = bundle["costs"].get<Matrix>();

        auto weights = bundle["outputActivationWeights"].get<Matrix>();

        auto normalizedWeights = softmax(weights);

        auto flattenedWeights = flatten(normalizedWeights);

        Dimension broadcastDimensions = removeDimensions(range(costs.size()),
            {costs.size().size() - 2});

        broadcast(costs, costs, flattenedWeights, broadcastDimensions, matrix::Multiply());

        bundle["weightedCosts"] = copy(costs);

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

        auto& outputActivationWeights = bundle["outputActivationWeights"].get<Matrix>();

        auto weights = flatten(softmax(outputActivationWeights));

        auto& outputDeltas = outputDeltasVector.front();

        //size_t beamSize      = outputActivationWeights.size()[0];
        size_t miniBatchSize = outputActivationWeights.size()[1];

        Dimension broadcastDimensions = removeDimensions(range(outputDeltas.size()),
            {outputDeltas.size().size() - 2});

        broadcast(outputDeltas, outputDeltas, weights, broadcastDimensions, matrix::Multiply());

        apply(outputDeltas, outputDeltas, matrix::Multiply(miniBatchSize));

        if(util::isLogEnabled("CostFunction::Detail"))
        {
            util::log("CostFunction::Detail") << " weighted output deltas : "
                << outputDeltas.debugString();
        }

        // compute weight deltas
        auto& weightedCosts = bundle["weightedCosts"].get<Matrix>();

        auto sum = reduce(apply(matrix::Matrix(weightedCosts), outputDeltas, matrix::Multiply()),
            {0}, matrix::Add());

        auto weightDeltas = apply(broadcast(outputDeltas, sum, {0}, matrix::Subtract()),
            weightedCosts, matrix::Multiply());

        auto& costs = bundle["costs"].get<Matrix>();

        apply(weightDeltas, weightDeltas, costs, matrix::Multiply());

        if(util::isLogEnabled("CostFunction::Detail"))
        {
            util::log("CostFunction::Detail") << " weight deltas : " << weightDeltas.debugString();
        }

        bundle["outputActivationWeightDeltas"] = weightDeltas;

    }
}

}

}


