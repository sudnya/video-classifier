/*	\file   SampleStatisticsEngine.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the SampleStatisticsEngine class.
*/

// Minerva Includes
#include <minerva/engine/interface/SampleStatisticsEngine.h>

#include <minerva/model/interface/Model.h>

#include <minerva/results/interface/ResultVector.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cmath>

namespace minerva
{

namespace engine
{

SampleStatisticsEngine::SampleStatisticsEngine()
{
	setEpochs(1);
}

SampleStatisticsEngine::~SampleStatisticsEngine()
{

}

void SampleStatisticsEngine::registerModel()
{
	_mean = 0.0;
	_standardDeviation = 0.0;
	_samples = 0.0;
	_sumOfSquaresOfDifferences = 0.0;
	
	_model->setAttribute("InputSampleMean",              0.0);
	_model->setAttribute("InputSampleStandardDeviation", 1.0);
}

void SampleStatisticsEngine::closeModel()
{
	_standardDeviation = std::sqrt(_sumOfSquaresOfDifferences / (_samples - 1.0));
	
	_model->setAttribute("InputSampleMean",              _mean);
	_model->setAttribute("InputSampleStandardDeviation", _standardDeviation);

	// TODO 
	// saveModel();
}
	
SampleStatisticsEngine::ResultVector SampleStatisticsEngine::runOnBatch(Matrix&& input, Matrix&& reference)
{
	util::log("SimpleStatisticsEngine") << "Computing sample statistics over " << input.size()[1] <<  " images...\n";
	
	for(auto element : input)
	{
		_samples += 1.0f;

		float delta = element - _mean;
		
		_mean = _mean + delta / _samples;
		
		_sumOfSquaresOfDifferences += delta * (element - _mean);
	}
	
	return ResultVector();
}

}

}


