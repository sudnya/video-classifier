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

    /*
    if(bundle.contains("outputActivationWeights"))
    {
        auto& costs = bundle["costs"].get<Matrix>();

        auto weights = flatten(bundle["outputActivationWeights"].get<Matrix>());

        Dimension broadcastDimensions = removeDimensions(range(costs.size()),
            {costs.size().size() - 2});

        broadcast(costs, costs, weights, broadcastDimensions, matrix::Multiply());

        bundle["weightedCosts"] = copy(costs);

        if(util::isLogEnabled("CostFunction::Detail"))
        {
            util::log("CostFunction::Detail") << " weighted costs : "
                << costs.debugString();
        }
    }*/
}

void CostFunction::computeDelta(Bundle& bundle) const
{
    computeDeltaImplementation(bundle);

    /*
    if(bundle.contains("outputActivationWeights"))
    {
        auto& outputDeltasVector = bundle["outputDeltas"].get<MatrixVector>();

        auto weights = flatten(bundle["outputActivationWeights"].get<Matrix>());

        auto& outputDeltas = outputDeltasVector.front();

        Dimension broadcastDimensions = removeDimensions(range(outputDeltas.size()),
            {outputDeltas.size().size() - 2});

        broadcast(outputDeltas, outputDeltas, weights, broadcastDimensions, matrix::Multiply());

        if(util::isLogEnabled("CostFunction::Detail"))
        {
            util::log("CostFunction::Detail") << " weighted output deltas : "
                << outputDeltas.debugString();
        }
    }*/
}

}

}


