/*	\file   Engine.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Engine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/Engine.h>

#include <minerva/results/interface/NullResultProcessor.h>
#include <minerva/results/interface/ResultVector.h>

#include <minerva/input/interface/InputDataProducerFactory.h>
#include <minerva/input/interface/InputDataProducer.h>

#include <minerva/model/interface/Model.h>

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
	setResultProcessor(new results::NullResultProcessor);
}

Engine::~Engine()
{

}

void Engine::setModel(Model* model)
{
	_model.reset(model);
}

void Engine::loadModel(const std::string& pathToModelFile)
{
	util::log("Engine") << "Loading model file '" << pathToModelFile
		<<  "'...\n";
	
	_model.reset(new Model(pathToModelFile));

	util::log("Engine") << " model loaded.\n";
}

void Engine::runOnDatabaseFile(const std::string& path)
{
	_model->load();

	registerModel();
		
	if(path.empty())
	{
		throw std::runtime_error("No input path provided.");
	}
	
	_dataProducer.reset(input::InputDataProducerFactory::createForDatabase(path));
	
	while(!_dataProducer->empty())
	{
		auto dataAndReference = std::move(_dataProducer->pop());
		
		auto results = runOnBatch(std::move(dataAndReference.first), std::move(dataAndReference.second));
		
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

neuralnetwork::NeuralNetwork Engine::getAggregateNetwork()
{
	neuralnetwork::NeuralNetwork network;
	
	auto& featureSelector = _model->getNeuralNetwork("FeatureSelector");
	auto& classifier      = _model->getNeuralNetwork("Classifier");
	
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
	auto& classifier      = _model->getNeuralNetwork("Classifier");
	
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


