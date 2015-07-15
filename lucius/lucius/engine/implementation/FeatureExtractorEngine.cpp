/*    \file   FeatureExtractorEngine.cpp
    \date   Saturday January 18, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the FeatureExtractorEngine class.
*/

// Lucious Includes
#include <lucious/engine/interface/FeatureExtractorEngine.h>

#include <lucious/network/interface/NeuralNetwork.h>

#include <lucious/results/interface/FeatureResultProcessor.h>
#include <lucious/results/interface/FeatureResult.h>

#include <lucious/results/interface/ResultVector.h>

#include <lucious/model/interface/Model.h>

#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixTransformations.h>

// Standard Library Includes
#include <cassert>

namespace lucious
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

