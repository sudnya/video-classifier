/*    \file   LearnerEngine.cpp
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LearnerEngine class.
*/

// Lucius Includes
#include <lucius/engine/interface/LearnerEngine.h>

#include <lucius/network/interface/NeuralNetwork.h>

#include <lucius/results/interface/ResultVector.h>
#include <lucius/results/interface/CostResult.h>

#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace engine
{

LearnerEngine::LearnerEngine()
{

}

LearnerEngine::~LearnerEngine()
{

}

void LearnerEngine::closeModel()
{

}

LearnerEngine::ResultVector LearnerEngine::runOnBatch(Bundle& bundle)
{
    auto network = getAggregateNetwork();

    network->setIsTraining(true);

    network->train(bundle);

    double cost = bundle["cost"].get<double>();

    restoreAggregateNetwork();

    ResultVector results;

    results.push_back(new results::CostResult(cost, getIteration()));

    return results;
}

bool LearnerEngine::requiresLabeledData() const
{
    return true;
}

}

}




