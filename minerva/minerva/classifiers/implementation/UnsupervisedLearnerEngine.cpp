/*	\file   UnsupervisedLearnerEngine.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the UnsupervisedLearnerEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/UnsupervisedLearnerEngine.h>
#include <minerva/classifiers/interface/UnsupervisedLearner.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

namespace minerva
{

namespace classifiers
{

UnsupervisedLearnerEngine::UnsupervisedLearnerEngine()
: _layersPerIteration(9)
{
	_layersPerIteration = util::KnobDatabase::getKnobValue<size_t>(
		"UnsupervisedLearnerEngine::LayersPerIteration", 9);
}

UnsupervisedLearnerEngine::~UnsupervisedLearnerEngine()
{

}

void UnsupervisedLearnerEngine::setLayersPerIteration(size_t l)
{
	_layersPerIteration = l;
}

void UnsupervisedLearnerEngine::runOnBatch(Matrix&& input)
{
	util::log("UnsupervisedLearnerEngine") << "Performing unsupervised "
		"learning on " << input.rows() <<  " samples...\n";
	
	auto totalLayers = _getTotalLayers();
	
	auto inputReference = input.add(1.0f).multiply(0.4f).add(0.1f);
	
	auto layerInput = std::move(input);

	for(size_t counter = 0; counter < totalLayers; counter += _layersPerIteration)
	{
		unsigned int counterEnd = std::min(counter + _layersPerIteration,
			totalLayers);

		util::log("UnsupervisedLearner") << "Training feature selector layers "
			<< counter << " to " << counterEnd << "\n";
		
		auto network = _formAugmentedNetwork(counter, counterEnd);
		
		network.train(layerInput, inputReference);
		
		if(counter + _layersPerIteration < totalLayers)
		{
			layerInput = copy.runInputs(layerInput);
		}
		
		_restoreAugmentedNetwork(network, counter);
	}
	
	util::log("UnsupervisedLearnerEngine") << " unsupervised "
		"learning finished, updating model.\n";
}

size_t UnsupervisedLearnerEngine::_getTotalLayers() const
{
	return _model->getNeuralNetwork("FeatureSelector").size();
}

neuralnetwork::NeuralNetwork UnsupervisedLearnerEngine::_formAugmentedNetwork(size_t layerBegin, size_t layerEnd)
{
	// Move the network into the temporary
	auto& featureSelector = model->getNeuralNetwork("FeatureSelector");
	
	auto inputCount = featureSelector.getInputCount();
	
	neuralnetwork::NeuralNetwork network;
	
	for(size_t layerId = layerBegin; layerId < layerEnd; ++layerId)
	{	
		network.addLayer(std::move(featureSelector[layerId]));
	}
	
	network.setUseSparseCostFunction(featureSelector.isUsingSparseCostFunction());
	
	// Create or restore the augmentor layers
	auto& augmentor = _getOrCreateAugmentor("FeatureSelector", layerBegin, network);
	
	// Merge the augmentor layers into the new network
	for(size_t layerId = 0, augmentorLayers = augmentor.size(); layerId < augmentorLayers; ++layerId)
	{
		network.addLayer(std::move(augmentor[layerId]));
	}

	return network;
}

void UnsupervisedLearnerEngine::_restoreAugmentedNetwork(neuralnetwork::NeuralNetwork& network, size_t layerBegin)
{
	auto& featureSelector = model->getNeuralNetwork("FeatureSelector");
	auto& augmentor = getAugmentor("FeatureSelector", layerBegin);

	size_t layerEnd = layerBegin + network.size() - augmentor.size();
	
	// restore the network layers
	for(size_t layerId = layerBegin; layerId < layerEnd; ++layerId)
	{	
		featureSelector[layerId] = std::move(network[layerId]);
	}
	
	// restore the augmentor layers
	for(size_t layerId = 0, lastLayer = augmentor.size(); layerId < lastLayer; ++layerId)
	{
		augmentor[layerId] = std::move(network[layerId + layerEnd]);
	}
}

neuralnetwork::NeuralNetwork& UnsupervisedLearnerEngine::_getOrCreateAugmentor(const std::string& name,
	size_t layer, neuralnetwork::NeuralNetwork& network)
{
	std::stringstream stream;
	
	stream << name << layer;
	
	auto augmentor = _augmentorNetworks.find(stream.str());
	
	if(augmentor == _augmentorNetworks.end())
	{
		augmentor = _augmentorNetworks.insert(std::make_pair(stream.str(),
			neuralnetwork::NeuralNetwork())).first;
		
		augmentor->mirror(network.front());
	}
	
	return augmentor->second;
}

neuralnetwork::NeuralNetwork& UnsupervisedLearnerEngine::_getAugmentor(const std::string& name,
	size_t layer)
{
	std::stringstream stream;
	
	stream << name << layer;
	
	auto augmentor = _augmentorNetworks.find(stream.str());
	
	assert(augmentor != _augmentorNetworks.end());
	
	return augmentor->second;
}

void UnsupervisedLearnerEngine::closeModel()
{
	saveModel();
}

}

}




