/*	\file   Engine.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Engine class.
*/

// Minerva Includes
#include <minerva/engine/interface/Engine.h>

#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/Layer.h>

#include <minerva/results/interface/ResultProcessorFactory.h>
#include <minerva/results/interface/ResultVector.h>
#include <minerva/results/interface/ResultProcessor.h>

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

namespace engine
{

Engine::Engine()
: _dataProducer(input::InputDataProducerFactory::create()), _resultProcessor(results::ResultProcessorFactory::create("NullResultProcessor"))
{
	
}

Engine::~Engine()
{

}

void Engine::setModel(Model* model)
{
	_model = model;
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
	
	_dataProducer->initialize();

	util::log("Engine") << "Running for " << _dataProducer->getEpochs() <<  " epochs.\n";
	for(size_t epoch = 0; epoch != _dataProducer->getEpochs(); ++epoch)
	{
		while(!_dataProducer->empty())
		{
			auto dataAndReference = std::move(_dataProducer->pop());
			
			auto results = runOnBatch(std::move(dataAndReference.first),
				std::move(dataAndReference.second));
			
			_resultProcessor->process(std::move(results));
		}
			
		util::log("Engine") << " Finished epoch " << epoch <<  ".\n";
		
		_dataProducer->reset();
	}

	// close
	closeModel();
}

void Engine::setOutputFilename(const std::string& filename)
{
	_resultProcessor->setOutputFilename(filename);
}

Engine::ResultProcessor* Engine::extractResultProcessor()
{
	return _resultProcessor.release();
}

Engine::Model* Engine::getModel()
{
	return _model;
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

network::NeuralNetwork* Engine::getAggregateNetwork()
{
	if(!_aggregateNetwork)
	{
		_aggregateNetwork.reset(new NeuralNetwork);
	}
	
	auto& featureSelector = _model->getNeuralNetwork("FeatureSelector");
	auto& classifier      = _model->getNeuralNetwork("Classifier");
	
	for(auto& layer : featureSelector)
	{
		_aggregateNetwork->addLayer(layer.release());
	}
	
	for(auto& layer : classifier)
	{
		_aggregateNetwork->addLayer(layer.release());
	}
	
	return _aggregateNetwork.get();
}

void Engine::restoreAggregateNetwork()
{
	assert(_aggregateNetwork);

	auto& featureSelector = _model->getNeuralNetwork("FeatureSelector");
	auto& classifier      = _model->getNeuralNetwork("Classifier");
	
	size_t layerId = 0;
	
	for(auto& layer : featureSelector)
	{
		layer = std::move((*_aggregateNetwork)[layerId]);
		++layerId;
	}
	
	for(auto& layer : classifier)
	{
		layer = std::move((*_aggregateNetwork)[layerId]);
		++layerId;
	}
	
	_aggregateNetwork->clear();
}

void Engine::_setupProducer(const std::string& path)
{
	std::unique_ptr<InputDataProducer> newProducer(input::InputDataProducerFactory::createForDatabase(path));
	
	newProducer->setRequiresLabeledData(requiresLabeledData());
	newProducer->setEpochs(_dataProducer->getEpochs());
	newProducer->setMaximumSamplesToRun(_dataProducer->getMaximumSamplesToRun());
	newProducer->setBatchSize(_dataProducer->getBatchSize());
	newProducer->setModel(_model);

	_dataProducer = std::move(newProducer);
	
}


}

}


