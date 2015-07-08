/*! \file   ResultProcessorFactory.cpp
	\date   Sunday January 11, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ResultProcessorfactor class.
*/

// Lucious Includes
#include <lucious/results/interface/ResultProcessorFactory.h>

#include <lucious/results/interface/NullResultProcessor.h>
#include <lucious/results/interface/LabelResultProcessor.h>
#include <lucious/results/interface/LabelMatchResultProcessor.h>
#include <lucious/results/interface/FeatureResultProcessor.h>

namespace lucious
{

namespace results
{

ResultProcessor* ResultProcessorFactory::create(const std::string& name)
{
	if(name == "NullResultProcessor")
	{
		return new NullResultProcessor;
	}
	else if(name == "LabelResultProcessor")
	{
		return new LabelResultProcessor;
	}
	else if(name == "LabelMatchResultProcessor")
	{
		return new LabelMatchResultProcessor;
	}
	else if(name == "FeatureResultProcessor")
	{
		return new FeatureResultProcessor;
	}
	
	return nullptr;
}

ResultProcessor* ResultProcessorFactory::create()
{
	return create("NullResultProcessor");
}

}

}



