/*! \file   ResultProcessorFactory.cpp
    \date   Sunday January 11, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ResultProcessorfactor class.
*/

// Lucius Includes
#include <lucius/results/interface/ResultProcessorFactory.h>

#include <lucius/results/interface/NullResultProcessor.h>
#include <lucius/results/interface/LabelResultProcessor.h>
#include <lucius/results/interface/LabelMatchResultProcessor.h>
#include <lucius/results/interface/GraphemeMatchResultProcessor.h>
#include <lucius/results/interface/FeatureResultProcessor.h>
#include <lucius/results/interface/VideoDisplayResultProcessor.h>
#include <lucius/results/interface/CostLoggingResultProcessor.h>

#include <lucius/util/interface/ParameterPack.h>

namespace lucius
{

namespace results
{

ResultProcessor* ResultProcessorFactory::create(const std::string& name)
{
    return create(name, util::ParameterPack());
}


ResultProcessor* ResultProcessorFactory::create(const std::string& name,
    const util::ParameterPack& parameters)
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
    else if(name == "GraphemeMatchResultProcessor")
    {
        return new GraphemeMatchResultProcessor;
    }
    else if(name == "FeatureResultProcessor")
    {
        return new FeatureResultProcessor;
    }
    else if(name == "VideoDisplayResultProcessor")
    {
        return new VideoDisplayResultProcessor;
    }
    else if(name == "CostLoggingResultProcessor")
    {
        auto outputPath = parameters.get<std::string>("OutputPath", "");

        return new CostLoggingResultProcessor(outputPath);
    }

    return nullptr;
}

ResultProcessor* ResultProcessorFactory::create()
{
    return create("NullResultProcessor");
}

}

}



