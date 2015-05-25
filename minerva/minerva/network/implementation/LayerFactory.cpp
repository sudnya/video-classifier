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
	if("FeedForwardLayer" == name)
	{
		return std::make_unique<FeedForwardLayer>();
	}
	else if ("RecurrentLayer" == name)
	{
		return std::make_unique<RecurrentLayer>();
	}
	else if ("ConvolutionalLayer" == name)
	{
		return std::make_unique<ConvolutionalLayer>();
	}

	return nullptr;
}

}

}


