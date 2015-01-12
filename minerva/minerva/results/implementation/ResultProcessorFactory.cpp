/*! \file   ResultProcessorFactory.cpp
	\date   Sunday January 11, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ResultProcessorfactor class.
*/

// Minerva Includes
#include <minerva/results/interface/ResultProcessorFactory.h>

#include <minerva/results/interface/NullResultProcessor.h>
#include <minerva/results/interface/LabelResultProcessor.h>
#include <minerva/results/interface/LabelMatchResultProcessor.h>
#include <minerva/results/interface/FeatureResultProcessor.h>

namespace minerva
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



