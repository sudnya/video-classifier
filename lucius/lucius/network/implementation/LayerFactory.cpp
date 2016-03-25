/*  \file   LayerFactory.cpp
    \date   November 12, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LayerFactory class
*/

// Lucius Includes
#include <lucius/network/interface/LayerFactory.h>

#include <lucius/network/interface/FeedForwardLayer.h>
#include <lucius/network/interface/RecurrentLayer.h>
#include <lucius/network/interface/ConvolutionalLayer.h>
#include <lucius/network/interface/AudioConvolutionalLayer.h>
#include <lucius/network/interface/BatchNormalizationLayer.h>
#include <lucius/network/interface/MaxPoolingLayer.h>
#include <lucius/network/interface/SubgraphLayer.h>
#include <lucius/network/interface/ActivationFunctionFactory.h>

#include <lucius/matrix/interface/Dimension.h>
#include <lucius/matrix/interface/Precision.h>

#include <lucius/util/interface/memory.h>

namespace lucius
{

namespace network
{

std::unique_ptr<Layer> LayerFactory::create(const std::string& name)
{
    return create(name, ParameterPack());
}

std::unique_ptr<Layer> LayerFactory::create(const std::string& name,
    const ParameterPack& parameters)
{
    size_t inputSizeHeight    = parameters.get("InputSizeHeight",   1);
    size_t inputSizeWidth     = parameters.get("InputSizeWidth",    1);
    size_t inputSizeChannels  = parameters.get("InputSizeChannels", 1);
    size_t inputSizeBatch     = parameters.get("InputSizeBatch",    1);

    size_t inputSizeAggregate = parameters.get("InputSizeAggregate",
        inputSizeHeight * inputSizeWidth * inputSizeChannels);

    std::unique_ptr<Layer> layer;

    if("FeedForwardLayer" == name)
    {
        size_t inputSize  = parameters.get("InputSize",  inputSizeAggregate);
        size_t outputSize = parameters.get("OutputSize", inputSize);

        auto precision = *matrix::Precision::fromString(parameters.get("Precision",
            matrix::Precision::getDefaultPrecision().toString()));

        layer = std::make_unique<FeedForwardLayer>(inputSize, outputSize, precision);
    }
    else if("RecurrentLayer" == name)
    {
        size_t size      = parameters.get("Size",      inputSizeAggregate);
        size_t batchSize = parameters.get("BatchSize", 1);

        auto precision = *matrix::Precision::fromString(parameters.get("Precision",
            matrix::Precision::getDefaultPrecision().toString()));

        layer = std::make_unique<RecurrentLayer>(size, batchSize, precision);
    }
    else if("AudioConvolutionalLayer" == name)
    {
        size_t inputSamples   = parameters.get("InputSamples",   inputSizeWidth);
        size_t inputTimesteps = parameters.get("InputTimesteps", inputSizeHeight);
        size_t inputChannels  = parameters.get("InputChannels",  inputSizeChannels);
        size_t inputBatch     = parameters.get("BatchSize",      inputSizeBatch);

        size_t filterSamples   = parameters.get("FilterSamples",   1);
        size_t filterTimesteps = parameters.get("FilterTimesteps", 1);
        size_t filterInputs    = parameters.get("FilterInputs",    inputChannels);
        size_t filterOutputs   = parameters.get("FilterOutputs",   1);

        size_t strideSamples   = parameters.get("StrideSamples",   1);
        size_t strideTimesteps = parameters.get("StrideTimesteps", 1);

        size_t paddingSamples   = parameters.get("PaddingSamples",   0);
        size_t paddingTimesteps = parameters.get("PaddingTimesteps", 0);

        auto precision = *matrix::Precision::fromString(parameters.get("Precision",
            matrix::Precision::getDefaultPrecision().toString()));

        layer = std::make_unique<AudioConvolutionalLayer>(
            matrix::Dimension({inputSamples,   inputTimesteps,  inputChannels,  inputBatch, 1}),
            matrix::Dimension({filterSamples,  filterTimesteps, filterInputs,   filterOutputs}),
            matrix::Dimension({strideSamples,  strideTimesteps}),
            matrix::Dimension({paddingSamples, paddingTimesteps}),
            precision
        );
    }
    else if("ConvolutionalLayer" == name)
    {
        size_t inputWidth  = parameters.get("InputWidth",  inputSizeWidth);
        size_t inputHeight = parameters.get("InputHeight", inputSizeHeight);
        size_t inputColors = parameters.get("InputColors", inputSizeChannels);
        size_t inputBatch  = parameters.get("BatchSize",   inputSizeBatch);

        size_t filterWidth   = parameters.get("FilterWidth",   1);
        size_t filterHeight  = parameters.get("FilterHeight",  1);
        size_t filterInputs  = parameters.get("FilterInputs",  inputColors);
        size_t filterOutputs = parameters.get("FilterOutputs", 1);

        size_t strideWidth  = parameters.get("StrideWidth",  1);
        size_t strideHeight = parameters.get("StrideHeight", 1);

        size_t paddingWidth  = parameters.get("PaddingWidth",  0);
        size_t paddingHeight = parameters.get("PaddingHeight", 0);

        auto precision = *matrix::Precision::fromString(parameters.get("Precision",
            matrix::Precision::getDefaultPrecision().toString()));

        layer = std::make_unique<ConvolutionalLayer>(
            matrix::Dimension(inputWidth,  inputHeight,  inputColors,  inputBatch, 1),
            matrix::Dimension({filterWidth, filterHeight, filterInputs, filterOutputs}),
            matrix::Dimension({strideWidth, strideHeight}),
            matrix::Dimension({paddingWidth, paddingHeight}),
            precision
        );
    }
    else if("BatchNormalizationLayer" == name)
    {
        size_t inputWidth  = parameters.get("InputWidth",  inputSizeWidth);
        size_t inputHeight = parameters.get("InputHeight", inputSizeHeight);
        size_t inputColors = parameters.get("InputColors", inputSizeChannels);
        size_t inputBatch  = parameters.get("BatchSize",   inputSizeBatch);

        auto precision = *matrix::Precision::fromString(parameters.get("Precision",
            matrix::Precision::getDefaultPrecision().toString()));

        layer = std::make_unique<BatchNormalizationLayer>(
            matrix::Dimension({inputWidth, inputHeight, inputColors, inputBatch, 1}), precision);
    }
    else if("MaxPoolingLayer" == name)
    {
        size_t inputWidth  = parameters.get("InputWidth",  inputSizeWidth);
        size_t inputHeight = parameters.get("InputHeight", inputSizeHeight);
        size_t inputColors = parameters.get("InputColors", inputSizeChannels);
        size_t inputBatch  = parameters.get("BatchSize",   inputSizeBatch);

        size_t width  = parameters.get("FilterWidth", 1);
        size_t height = parameters.get("FilterHeight", 1);

        auto precision = *matrix::Precision::fromString(parameters.get("Precision",
            matrix::Precision::getDefaultPrecision().toString()));

        layer = std::make_unique<MaxPoolingLayer>(
            matrix::Dimension({inputWidth, inputHeight, inputColors, inputBatch, 1}),
            matrix::Dimension({width, height}), precision);
    }
    else if("SubgraphLayer" == name)
    {
        layer = std::make_unique<SubgraphLayer>();
    }

    if(layer && parameters.contains("ActivationFunction"))
    {
        layer->setActivationFunction(ActivationFunctionFactory::create(
            parameters.get<std::string>("ActivationFunction")));
    }

    return layer;
}

}

}


