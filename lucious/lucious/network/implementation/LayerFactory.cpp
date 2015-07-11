/*  \file   LayerFactory.cpp
    \date   November 12, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LayerFactory class
*/

// Lucious Includes
#include <lucious/network/interface/LayerFactory.h>

#include <lucious/network/interface/FeedForwardLayer.h>
#include <lucious/network/interface/RecurrentLayer.h>
#include <lucious/network/interface/ConvolutionalLayer.h>

#include <lucious/matrix/interface/Dimension.h>

#include <lucious/util/interface/memory.h>

namespace lucious
{

namespace network
{

std::unique_ptr<Layer> LayerFactory::create(const std::string& name)
{
    return create(name, ParameterPack());
}

std::unique_ptr<Layer> LayerFactory::create(const std::string& name, const ParameterPack& parameters)
{
    if("FeedForwardLayer" == name)
    {
        size_t inputSize  = parameters.get("InputSize",  1);
        size_t outputSize = parameters.get("OutputSize", inputSize);

        return std::make_unique<FeedForwardLayer>(inputSize, outputSize);
    }
    else if("RecurrentLayer" == name)
    {
        size_t size      = parameters.get("Size",      1);
        size_t batchSize = parameters.get("BatchSize", 1);

        return std::make_unique<RecurrentLayer>(size, batchSize);
    }
    else if("ConvolutionalLayer" == name)
    {
        size_t inputWidth  = parameters.get("InputWidth",  1);
        size_t inputHeight = parameters.get("InputHeight", 1);
        size_t inputColors = parameters.get("InputColors", 1);
        size_t inputBatch  = parameters.get("BatchSize",   1);

        size_t filterWidth   = parameters.get("FilterWidth",   1);
        size_t filterHeight  = parameters.get("FilterHeight",  1);
        size_t filterInputs  = parameters.get("FilterInputs",  1);
        size_t filterOutputs = parameters.get("FilterOutputs", 1);

        size_t strideWidth  = parameters.get("StrideWidth",  1);
        size_t strideHeight = parameters.get("StrideHeight", 1);

        size_t paddingWidth  = parameters.get("PaddingWidth",  0);
        size_t paddingHeight = parameters.get("PaddingHeight", 0);

        return std::make_unique<ConvolutionalLayer>(
            matrix::Dimension(inputWidth,  inputHeight,  inputColors,  inputBatch),
            matrix::Dimension({filterWidth, filterHeight, filterInputs, filterOutputs}),
            matrix::Dimension({strideWidth, strideHeight}),
            matrix::Dimension({paddingWidth, paddingHeight})
        );
    }

    return nullptr;
}

}

}


