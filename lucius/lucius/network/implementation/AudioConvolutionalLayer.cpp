/*  \file   AudioConvolutionalLayer.h
    \author Gregory Diamos
    \date   Dec 19, 2015
    \brief  The source file for the AudioConvolutionalLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/AudioConvolutionalLayer.h>
#include <lucius/network/interface/ConvolutionalLayer.h>

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

AudioConvolutionalLayer::AudioConvolutionalLayer()
: AudioConvolutionalLayer({}, {}, {}, {}, matrix::Precision::getDefaultPrecision())
{

}

AudioConvolutionalLayer::~AudioConvolutionalLayer()
{

}

AudioConvolutionalLayer::AudioConvolutionalLayer(const matrix::Dimension& inputSize,
    const matrix::Dimension& filterSize,
    const matrix::Dimension& filterStride,
    const matrix::Dimension& inputPadding)
: AudioConvolutionalLayer(inputSize, filterSize, filterStride, inputPadding,
    matrix::Precision::getDefaultPrecision())
{

}

AudioConvolutionalLayer::AudioConvolutionalLayer(const matrix::Dimension& inputSize,
    const matrix::Dimension& filterSize,
    const matrix::Dimension& filterStride,
    const matrix::Dimension& inputPadding,
    const matrix::Precision& precision)
: _layer(std::make_unique<ConvolutionalLayer>(
    inputSize, filterSize, filterStride, inputPadding, precision))
{

}

AudioConvolutionalLayer::AudioConvolutionalLayer(const AudioConvolutionalLayer& l)
: _layer(std::make_unique<ConvolutionalLayer>(*l._layer))
{

}

AudioConvolutionalLayer& AudioConvolutionalLayer::operator=(const AudioConvolutionalLayer& l)
{
    Layer::operator=(l);

    *_layer = *l._layer;

    return *this;
}

void AudioConvolutionalLayer::initialize()
{
    _layer->initialize();
}

void AudioConvolutionalLayer::popReversePropagationData()
{
    _layer->popReversePropagationData();
}

void AudioConvolutionalLayer::clearReversePropagationData()
{
    _layer->clearReversePropagationData();
}

void AudioConvolutionalLayer::setShouldComputeDeltas(bool shouldComputeDeltas)
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

void AudioConvolutionalLayer::runForwardImplementation(MatrixVector& outputActivations,
    const MatrixVector& inputActivations)
{
    MatrixVector storage;

    storage.push_back(permuteDimensionsForward(inputActivations.back()));

    _layer->runForwardImplementation(outputActivations, storage);

    outputActivations.back() = permuteDimensionsBackward(outputActivations.back());

    if(util::isLogEnabled("AudioConvolutionalLayer::Detail"))
    {
        util::log("AudioConvolutionalLayer::Detail") << "  activation: "
            << outputActivations.back().debugString();
    }
    else
    {
        util::log("AudioConvolutionalLayer") << "  activation: "
            << outputActivations.back().shapeString() << "\n";
    }
}

void AudioConvolutionalLayer::runReverseImplementation(MatrixVector& gradients,
    MatrixVector& inputDeltas,
    const MatrixVector& outputDeltas)
{
    assert(outputDeltas.size() == 1);

    MatrixVector outputDeltasStorage;

    outputDeltasStorage.push_back(permuteDimensionsForward(outputDeltas.front()));

    _layer->runReverseImplementation(gradients, inputDeltas, outputDeltasStorage);

    inputDeltas.front() = permuteDimensionsBackward(inputDeltas.front());
}

MatrixVector& AudioConvolutionalLayer::weights()
{
    return _layer->weights();
}

const MatrixVector& AudioConvolutionalLayer::weights() const
{
    return _layer->weights();
}

const matrix::Precision& AudioConvolutionalLayer::precision() const
{
    return _layer->precision();
}

double AudioConvolutionalLayer::computeWeightCost() const
{
    return _layer->computeWeightCost();
}

Dimension AudioConvolutionalLayer::getInputSize() const
{
    return permuteDimensionsBackward(_layer->getInputSize());
}

Dimension AudioConvolutionalLayer::getOutputSize() const
{
    return permuteDimensionsBackward(_layer->getOutputSize());
}

size_t AudioConvolutionalLayer::getInputCount()  const
{
    return _layer->getInputCount();
}

size_t AudioConvolutionalLayer::getOutputCount() const
{
    return _layer->getOutputCount();
}

size_t AudioConvolutionalLayer::totalNeurons() const
{
    return _layer->totalNeurons();
}

size_t AudioConvolutionalLayer::totalConnections() const
{
    return _layer->totalConnections();
}

size_t AudioConvolutionalLayer::getFloatingPointOperationCount() const
{
    return _layer->getFloatingPointOperationCount();
}

size_t AudioConvolutionalLayer::getActivationMemory() const
{
    return _layer->getActivationMemory();
}

void AudioConvolutionalLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    _layer->save(archive, properties);
}

void AudioConvolutionalLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    _layer->load(archive, properties);
}

std::unique_ptr<Layer> AudioConvolutionalLayer::clone() const
{
    return std::unique_ptr<Layer>(new AudioConvolutionalLayer(*this));
}

std::string AudioConvolutionalLayer::getTypeName() const
{
    return "AudioConvolutionalLayer";
}

}

}


