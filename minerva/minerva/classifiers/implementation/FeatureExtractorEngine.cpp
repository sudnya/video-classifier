/*	\file   FeatureExtractorEngine.cpp
	\date   Saturday January 18, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the FeatureExtractorEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/FeatureExtractorEngine.h>

#include <minerva/results/interface/FeatureResultProcessor.h>
#include <minerva/results/interface/FeatureResult.h>

#include <minerva/results/interface/ResultVector.h>

#include <minerva/model/interface/Model.h>

#include <minerva/matrix/interface/Matrix.h>

// Standard Library Includes
#include <cassert>

namespace minerva
{

namespace classifiers
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
	size_t samples = features.rows();
	
	ResultVector result;
	
	for(size_t sample = 0; sample < samples; ++sample)
	{
		result.push_back(new results::FeatureResult(features.slice(sample, 0, 1, features.columns())));
	}
	
	return result;
}
	
}

}

