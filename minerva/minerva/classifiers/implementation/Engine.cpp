/*	\file   Engine.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Engine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/Engine.h>

#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/Layer.h>
#include <minerva/network/interface/CostFunction.h>

#include <minerva/optimizer/interface/NeuralNetworkSolver.h>

#include <minerva/results/interface/NullResultProcessor.h>
#include <minerva/results/interface/ResultVector.h>

#include <minerva/input/interface/InputDataProducerFactory.h>
#include <minerva/input/interface/InputDataProducer.h>

#include <minerva/model/interface/Model.h>

#include <minerva/matrix/interface/Matrix.h>

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

void Engine::loadModel(const std::string& pathToModelFile)
{
	util::log("Engine") << "Loading model file '" << pathToModelFile
		<<  "'...\n";
	
	_model.reset(new Model(pathToModelFile));

	util::log("Engine") << " model loaded.\n";
}

void Engine::setModel(Model* model)
{
	_model.reset(model);
}

void Engine::setResultProcessor(ResultProcessor* processor)
{
	_resultProcessor.reset(processor);
}

void Engine::runOnDatabaseFile(const std::string& path)
{
	_model->load();

	registerModel();
		
	if(path.empty())
	{
		throw std::runtime_error("No input path provided.");
	}
	
	_setupProducer(path);
	
	while(!_dataProducer->empty())
	{
		auto dataAndReference = std::move(_dataProducer->pop());
		
		auto results = runOnBatch(std::move(dataAndReference.first),
			std::move(dataAndReference.second));
		
		_resultProcessor->process(std::move(results));
	}

	// close
	closeModel();
}

void Engine::setOutputFilename(const std::string& filename)
{
	_resultProcessor->setOutputFilename(filename);
}

Engine::Model* Engine::extractModel()
{
	return _model.release();
}

Engine::ResultProcessor* Engine::extractResultProcessor()
{
	return _resultProcessor.release();
}

Engine::Model* Engine::getModel()
{
	return _model.get();
}

Engine::ResultProcessor* Engine::getResultProcessor()
{
	return _resultProcessor.get();
}

void Engine::setMaximumSamplesToRun(size_t samples)
{
	_dataProducer->setMaximumSamplesToRun(samples);
}

void Engine::setBatchSize(size_t samples)
{
	_dataProducer->setBatchSize(samples);
}

void Engine::setEpochs(size_t epochs)
{
	_dataProducer->setEpochs(epochs);
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

network::NeuralNetwork Engine::getAggregateNetwork()
{
	network::NeuralNetwork network;
	
	auto& featureSelector = _model->getNeuralNetwork("FeatureSelector");
	auto& classifier      = _model->getNeuralNetwork("Classifier");
	
	for(auto& layer : featureSelector)
	{
		network.addLayer(layer.release());
	}
	
	for(auto& layer : classifier)
	{
		network.addLayer(layer.release());
	}
	
	return network;
}

void Engine::restoreAggregateNetwork(network::NeuralNetwork& network)
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

void Engine::_setupProducer(const std::string& path)
{
	_dataProducer.reset(
		input::InputDataProducerFactory::createForDatabase(path));
	
	_dataProducer->setRequiresLabeledData(requiresLabeledData());
}


}

}


