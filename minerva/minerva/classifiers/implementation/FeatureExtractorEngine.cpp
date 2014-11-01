/*	\file   FeatureExtractorEngine.cpp
	\date   Saturday January 18, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the FeatureExtractorEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/FeatureExtractorEngine.h>

// Standard Library Includes
#include <cassert>

namespace minerva
{

namespace classifiers
{

FeatureExtractorEngine::FeatureExtractorEngine()
{
	setResultProcessor(new FeatureResultProcessor);
}

typedef std::vector<std::string> StringVector;
typedef matrix::Matrix Matrix;
typedef video::ImageVector ImageVector;

ResultVector FeatureExtractorEngine::runBatch(Matrix&& input, Matrix&& reference)
{
	auto& featureSelector = _model->getNeuralNetwork("FeatureSelector");
	
	auto features = featureSelector.runInputs(input);
	
	// convert to results
	size_t samples = features.rows();
	
	ResultVector results;
	
	for(size_t sample = 0; sample < samples; ++sample)
	{
		result.push_back(new FeatureResult(features.slice(sample, 0, 1, features.columns())));
	}
	
	return result;
}
	
}

}

