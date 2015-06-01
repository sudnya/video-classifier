/*  \file   LayerFactory.cpp
    \date   November 12, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LayerFactory class
*/

// Minerva Includes
#include <minerva/network/interface/LayerFactory.h>

#include <minerva/network/interface/FeedForwardLayer.h>
#include <minerva/network/interface/RecurrentLayer.h>
#include <minerva/network/interface/ConvolutionalLayer.h>

#include <minerva/util/interface/memory.h>

namespace minerva
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
        return std::make_unique<ConvolutionalLayer>();
    }

    return nullptr;
}

}

}


