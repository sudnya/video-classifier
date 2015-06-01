/*    \file   UnsupervisedLearnerEngine.cpp
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the UnsupervisedLearnerEngine class.
*/

// Minerva Includes
#include <minerva/engine/interface/UnsupervisedLearnerEngine.h>

#include <minerva/model/interface/Model.h>

#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/Layer.h>

#include <minerva/results/interface/ResultVector.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

namespace minerva
{

namespace engine
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

static size_t getTotalLayers(model::Model& model)
{
    return model.getNeuralNetwork("FeatureSelector").size();
}

UnsupervisedLearnerEngine::ResultVector UnsupervisedLearnerEngine::runOnBatch(Matrix&& input, Matrix&& reference)
{
    util::log("UnsupervisedLearnerEngine") << "Performing unsupervised "
        "learning on " << input.size()[1] <<  " samples...\n";

    auto totalLayers = getTotalLayers(*getModel());

    auto inputReference = matrix::apply(matrix::apply(matrix::apply(input, matrix::Add(1.0)), matrix::Multiply(0.4)), matrix::Add(0.1));

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
            layerInput = network.runInputs(layerInput);
        }

        _restoreAugmentedNetwork(network, counter);
    }

    util::log("UnsupervisedLearnerEngine") << " unsupervised "
        "learning finished, updating model.\n";

    return ResultVector();
}

network::NeuralNetwork UnsupervisedLearnerEngine::_formAugmentedNetwork(size_t layerBegin, size_t layerEnd)
{
    // Move the network into the temporary
    auto& featureSelector = getModel()->getNeuralNetwork("FeatureSelector");

    network::NeuralNetwork network;

    for(size_t layerId = layerBegin; layerId < layerEnd; ++layerId)
    {
        network.addLayer(std::move(featureSelector[layerId]));
    }

    // Create or restore the augmentor layers
    auto& augmentor = _getOrCreateAugmentor("FeatureSelector", layerBegin, network);

    // Merge the augmentor layers into the new network
    for(size_t layerId = 0, augmentorLayers = augmentor.size(); layerId < augmentorLayers; ++layerId)
    {
        network.addLayer(std::move(augmentor[layerId]));
    }

    return network;
}

void UnsupervisedLearnerEngine::_restoreAugmentedNetwork(network::NeuralNetwork& network, size_t layerBegin)
{
    auto& featureSelector = getModel()->getNeuralNetwork("FeatureSelector");
    auto& augmentor = _getAugmentor("FeatureSelector", layerBegin);

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

static void mirrorNeuralNetwork(network::NeuralNetwork& network)
{
    network.addLayer(network.front()->mirror());
}

network::NeuralNetwork& UnsupervisedLearnerEngine::_getOrCreateAugmentor(const std::string& name,
    size_t layer, network::NeuralNetwork& network)
{
    std::stringstream stream;

    stream << name << layer;

    auto augmentor = _augmentorNetworks.find(stream.str());

    if(augmentor == _augmentorNetworks.end())
    {
        augmentor = _augmentorNetworks.insert(std::make_pair(stream.str(),
            network::NeuralNetwork())).first;

        mirrorNeuralNetwork(augmentor->second);
    }

    return augmentor->second;
}

network::NeuralNetwork& UnsupervisedLearnerEngine::_getAugmentor(const std::string& name,
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




