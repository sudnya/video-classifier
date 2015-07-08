/*    \file   LearnerEngine.cpp
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LearnerEngine class.
*/

// Lucious Includes
#include <lucious/engine/interface/LearnerEngine.h>

#include <lucious/network/interface/NeuralNetwork.h>

#include <lucious/results/interface/ResultVector.h>

#include <lucious/matrix/interface/Matrix.h>

#include <lucious/util/interface/debug.h>

// Standard Library Includes
#include <cassert>

namespace lucious
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
    //saveModel();
}

LearnerEngine::ResultVector LearnerEngine::runOnBatch(Matrix&& input, Matrix&& reference)
{
    util::log("LearnerEngine") << "Performing supervised "
        "learning on batch of " << input.size()[input.size().size()-2] <<  " images...\n";

    auto network = getAggregateNetwork();

    network->train(std::move(input), std::move(reference));

    restoreAggregateNetwork();

    return ResultVector();
}

bool LearnerEngine::requiresLabeledData() const
{
    return true;
}

}

}




