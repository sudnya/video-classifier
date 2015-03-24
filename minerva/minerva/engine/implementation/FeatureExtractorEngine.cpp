/*	\file   FeatureExtractorEngine.cpp
	\date   Saturday January 18, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the FeatureExtractorEngine class.
*/

// Minerva Includes
#include <minerva/engine/interface/FeatureExtractorEngine.h>

#include <minerva/network/interface/NeuralNetwork.h>

#include <minerva/results/interface/FeatureResultProcessor.h>
#include <minerva/results/interface/FeatureResult.h>

#include <minerva/results/interface/ResultVector.h>

#include <minerva/model/interface/Model.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixOperations.h>

// Standard Library Includes
#include <cassert>

namespace minerva
{

namespace engine
{

FeatureExtractorEngine::FeatureExtractorEngine()
{
	setResultProcessor(new results::FeatureResultProcessor);
}

FeatureExtractorEngine::ResultVector FeatureExtractorEngine::runOnBatch(Matrix&& input, Matrix&& reference)
{
	auto& featureSelector = _model->getNeuralNetwork("FeatureSelector");
	
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

