/*    \file   FeatureExtractorEngine.cpp
    \date   Saturday January 18, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the FeatureExtractorEngine class.
*/

// Lucius Includes
#include <lucius/engine/interface/FeatureExtractorEngine.h>

#include <lucius/network/interface/NeuralNetwork.h>

#include <lucius/results/interface/FeatureResultProcessor.h>
#include <lucius/results/interface/FeatureResult.h>

#include <lucius/results/interface/ResultVector.h>

#include <lucius/model/interface/Model.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixTransformations.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace engine
{

FeatureExtractorEngine::FeatureExtractorEngine()
{
    setResultProcessor(new results::FeatureResultProcessor);
}

FeatureExtractorEngine::ResultVector FeatureExtractorEngine::runOnBatch(Matrix&& input, Matrix&& reference)
{
    auto& featureSelector = getModel()->getNeuralNetwork("FeatureSelector");

    auto features = featureSelector.runInputs(input);

    // convert to results
    size_t samples = features.size()[1];

    ResultVector result;

    for(size_t sample = 0; sample < samples; ++sample)
    {
        result.push_back(new results::FeatureResult(slice(features, {0, sample}, {features.size()[0], sample + 1})));
    }

    return result;
}

}

}

