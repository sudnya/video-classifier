/*  \file   AudioMaxPoolingLayer.h
    \author Gregory Diamos
    \date   Dec 19, 2015
    \brief  The source file for the AudioMaxPoolingLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/AudioMaxPoolingLayer.h>
#include <lucius/network/interface/MaxPoolingLayer.h>

#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Precision.h>
#include <lucius/matrix/interface/Dimension.h>

#include <lucius/matrix/interface/DimensionTransformations.h>
#include <lucius/matrix/interface/CopyOperations.h>

#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace network
{

typedef matrix::MatrixVector MatrixVector;
typedef matrix::Matrix Matrix;
typedef matrix::Dimension Dimension;

AudioMaxPoolingLayer::AudioMaxPoolingLayer()
: AudioMaxPoolingLayer({}, {}, matrix::Precision::getDefaultPrecision())
{

}

AudioMaxPoolingLayer::~AudioMaxPoolingLayer()
{

}

AudioMaxPoolingLayer::AudioMaxPoolingLayer(const matrix::Dimension& inputSize,
    const matrix::Dimension& filterSize)
: AudioMaxPoolingLayer(inputSize, filterSize,
    matrix::Precision::getDefaultPrecision())
{

}

AudioMaxPoolingLayer::AudioMaxPoolingLayer(const matrix::Dimension& inputSize,
    const matrix::Dimension& filterSize,
    const matrix::Precision& precision)
: _layer(std::make_unique<MaxPoolingLayer>(
    inputSize, filterSize, precision))
{

}

AudioMaxPoolingLayer::AudioMaxPoolingLayer(const AudioMaxPoolingLayer& l)
: _layer(std::make_unique<MaxPoolingLayer>(*l._layer))
{

}

AudioMaxPoolingLayer& AudioMaxPoolingLayer::operator=(const AudioMaxPoolingLayer& l)
{
    Layer::operator=(l);

    *_layer = *l._layer;

    return *this;
}

void AudioMaxPoolingLayer::initialize()
{
    _layer->initialize();
}

void AudioMaxPoolingLayer::popReversePropagationData()
{
    _layer->popReversePropagationData();
}

void AudioMaxPoolingLayer::clearReversePropagationData()
{
    _layer->clearReversePropagationData();
}

void AudioMaxPoolingLayer::setShouldComputeDeltas(bool shouldComputeDeltas)
{
    _layer->setShouldComputeDeltas(shouldComputeDeltas);
}

static Matrix permuteDimensionsForward(Matrix matrix)
{
    return permuteDimensions(matrix, {0, 4, 1, 2, 3});
}

static Matrix permuteDimensionsBackward(Matrix matrix)
{
    return permuteDimensions(matrix, {0, 2, 3, 4, 1});
}

static Dimension permuteDimensionsBackward(Dimension dimension)
{
    return selectDimensions(dimension, {0, 2, 3, 4, 1});
}

void AudioMaxPoolingLayer::runForwardImplementation(MatrixVector& outputActivations,
    const MatrixVector& inputActivations)
{
    MatrixVector storage;

    storage.push_back(permuteDimensionsForward(inputActivations.back()));

    _layer->runForwardImplementation(outputActivations, storage);

    outputActivations.back() = permuteDimensionsBackward(outputActivations.back());

    if(util::isLogEnabled("AudioMaxPoolingLayer::Detail"))
    {
        util::log("AudioMaxPoolingLayer::Detail") << "  activation: "
            << outputActivations.back().debugString();
    }
    else
    {
        util::log("AudioMaxPoolingLayer") << "  activation: "
            << outputActivations.back().shapeString() << "\n";
    }
}

void AudioMaxPoolingLayer::runReverseImplementation(MatrixVector& gradients,
    MatrixVector& inputDeltas,
    const MatrixVector& outputDeltas)
{
    assert(outputDeltas.size() == 1);

    MatrixVector outputDeltasStorage;

    outputDeltasStorage.push_back(permuteDimensionsForward(outputDeltas.front()));

    _layer->runReverseImplementation(gradients, inputDeltas, outputDeltasStorage);

    inputDeltas.front() = permuteDimensionsBackward(inputDeltas.front());
}

MatrixVector& AudioMaxPoolingLayer::weights()
{
    return _layer->weights();
}

const MatrixVector& AudioMaxPoolingLayer::weights() const
{
    return _layer->weights();
}

const matrix::Precision& AudioMaxPoolingLayer::precision() const
{
    return _layer->precision();
}

double AudioMaxPoolingLayer::computeWeightCost() const
{
    return _layer->computeWeightCost();
}

Dimension AudioMaxPoolingLayer::getInputSize() const
{
    return permuteDimensionsBackward(_layer->getInputSize());
}

Dimension AudioMaxPoolingLayer::getOutputSize() const
{
    return permuteDimensionsBackward(_layer->getOutputSize());
}

size_t AudioMaxPoolingLayer::getInputCount()  const
{
    return _layer->getInputCount();
}

size_t AudioMaxPoolingLayer::getOutputCount() const
{
    return _layer->getOutputCount();
}

size_t AudioMaxPoolingLayer::totalNeurons() const
{
    return _layer->totalNeurons();
}

size_t AudioMaxPoolingLayer::totalConnections() const
{
    return _layer->totalConnections();
}

size_t AudioMaxPoolingLayer::getFloatingPointOperationCount() const
{
    return _layer->getFloatingPointOperationCount();
}

size_t AudioMaxPoolingLayer::getActivationMemory() const
{
    return _layer->getActivationMemory();
}

void AudioMaxPoolingLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    _layer->save(archive, properties);
}

void AudioMaxPoolingLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    _layer->load(archive, properties);
}

std::unique_ptr<Layer> AudioMaxPoolingLayer::clone() const
{
    return std::unique_ptr<Layer>(new AudioMaxPoolingLayer(*this));
}

std::string AudioMaxPoolingLayer::getTypeName() const
{
    return "AudioMaxPoolingLayer";
}

}

}



