/*  \file   LayerFactory.cpp
    \date   November 12, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LayerFactory class
*/

// Minerva Includes
#include <minerva/network/interface/LayerFactory.h>

#include <minerva/network/interface/FeedForwardLayer.h>
#include <minerva/network/interface/RecurrentLayer.h>

namespace minerva
{

namespace network
{

Layer* LayerFactory::create(const std::string& name)
{
	if("FeedForwardLayer" == name)
	{
		return new FeedForwardLayer;
	}
	else if ("RecurrentLayer" == name)
	{
		return new RecurrentLayer;
	}

	return nullptr;
}

}

}


