/*	\file   Engine.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Engine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/Engine.h>

#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/database/interface/SampleDatabase.h>
#include <minerva/database/interface/Sample.h>

#include <minerva/video/interface/Image.h>
#include <minerva/video/interface/Video.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/math.h>
#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <stdexcept>
#include <fstream>
#include <random>
#include <cstdlib>
#include <algorithm>

namespace minerva
{

namespace classifiers
{

Engine::Engine()
{
	_maximumSamplesToRun = util::KnobDatabase::getKnobValue(
		"Engine::MaximumVideoFrames", 20000000);
	_batchSize = util::KnobDatabase::getKnobValue(
		"Engine::ImageBatchSize", 512);

	setResultProcessor(new NullResultProcessor);
}

Engine::~Engine()
{

}

void Engine::setModel(ClassificationModel* model)
{
	_model.reset(model);
}

void Engine::loadModel(const std::string& pathToModelFile)
{
	util::log("Engine") << "Loading model file '" << pathToModelFile
		<<  "'...\n";
	
	_model.reset(new ClassificationModel(pathToModelFile));

	util::log("Engine") << " model loaded.\n";
}

void Engine::runOnDatabaseFile(const std::string& path)
{
	_model->load();

	registerModel();
		
	if(paths.empty())
	{
		throw std::runtime_error("No input path provided.");
	}
	
	_dataProducer.reset(InputDataProducerFactory::createForDatabase(path));
	
	while(!_producer->empty())
	{
		auto results = runOnBatch(std::move(_producer->pop()));
		
		_resultProcessor->process(std::move(results));
	}

	// close
	closeModel();
}

std::string Engine::reportStatisticsString() const
{
	std::stringstream stream;

	reportStatistics(stream);
	
	return stream.str();
}

void Engine::reportStatistics(std::ostream& stream) const
{
	// intentionally blank
}

void Engine::registerModel()
{
	// intentionally blank
}

void Engine::closeModel()
{
	// intentionally blank
}
	
bool Engine::requiresLabeledData() const
{
	return false;
}

void Engine::saveModel()
{
	if(_model) _model->save();
}

neuralnetwork::NeuralNetwork Engine::getAggregateModel()
{
	neuralnetwork::NeuralNetwork network;
	
	auto& featureSelector = _model->getNeuralNetwork("FeatureSelector");
	auto& classifier      = _model->getNeuralNetwork("Classifiers");
	
	for(auto&& layer : featureSelector)
	{
		network.addLayer(std::move(layer));
	}
	
	for(auto&& layer : classifier)
	{
		network.addLayer(std::move(layer));
	}
	
	return network;
}

void Engine::restoreAggregateNetwork(neuralnetwork::NeuralNetwork& network)
{
	auto& featureSelector = _model->getNeuralNetwork("FeatureSelector");
	auto& classifier      = _model->getNeuralNetwork("Classifiers");
	
	size_t layerId = 0;
	
	for(auto& layer : featureSelector)
	{
		layer = std::move(network[layerId]);
		++layerId;
	}
	
	for(auto& layer : classifier)
	{
		layer = std::move(network[layerId]);
		++layerId;
	}
}

}

}


