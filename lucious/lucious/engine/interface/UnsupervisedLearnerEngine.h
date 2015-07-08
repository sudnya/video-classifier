/*    \file   UnsupervisedLearnerEngine.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the UnsupervisedLearnerEngine class.
*/

#pragma once

// Lucious Includes
#include <lucious/engine/interface/Engine.h>

// Standard Library Includes
#include <map>

namespace lucious
{

namespace engine
{


/*! \brief Performs unsupervised learning on a given model. */
class UnsupervisedLearnerEngine : public Engine
{
public:
    UnsupervisedLearnerEngine();
    virtual ~UnsupervisedLearnerEngine();

public:
    UnsupervisedLearnerEngine(const UnsupervisedLearnerEngine&) = delete;
    UnsupervisedLearnerEngine& operator=(const UnsupervisedLearnerEngine&) = delete;

public:
    void setLayersPerIteration(size_t layers);

private:
    virtual void closeModel();

private:
    virtual ResultVector runOnBatch(Matrix&& samples, Matrix&& reference);

private:
    NeuralNetwork _formAugmentedNetwork(size_t layerBegin, size_t layerEnd);
    void _restoreAugmentedNetwork(NeuralNetwork& network, size_t layerBegin);
    NeuralNetwork& _getOrCreateAugmentor(const std::string& name, size_t layer, NeuralNetwork& network);
    NeuralNetwork& _getAugmentor(const std::string& name, size_t layer);

private:
    size_t _layersPerIteration;

private:
    typedef std::map<std::string, NeuralNetwork> NetworkMap;

private:
    NetworkMap _augmentorNetworks;

};

}

}




