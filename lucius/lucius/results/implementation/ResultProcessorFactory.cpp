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

std::unique_ptr<ResultProcessor> ResultProcessorFactory::create(const std::string& name)
{
    return create(name, util::ParameterPack());
}


std::unique_ptr<ResultProcessor> ResultProcessorFactory::create(const std::string& name,
    const util::ParameterPack& parameters)
{
    if(name == "NullResultProcessor")
    {
        return std::make_unique<NullResultProcessor>();
    }
    else if(name == "LabelResultProcessor")
    {
        return std::make_unique<LabelResultProcessor>();
    }
    else if(name == "LabelMatchResultProcessor")
    {
        return std::make_unique<LabelMatchResultProcessor>();
    }
    else if(name == "GraphemeMatchResultProcessor")
    {
        return std::make_unique<GraphemeMatchResultProcessor>();
    }
    else if(name == "FeatureResultProcessor")
    {
        return std::make_unique<FeatureResultProcessor>();
    }
    else if(name == "VideoDisplayResultProcessor")
    {
        return std::make_unique<VideoDisplayResultProcessor>();
    }
    else if(name == "CostLoggingResultProcessor")
    {
        auto outputPath = parameters.get<std::string>("OutputPath", "");

        return std::make_unique<CostLoggingResultProcessor>(outputPath);
    }

    return std::unique_ptr<ResultProcessor>(nullptr);
}

std::unique_ptr<ResultProcessor> ResultProcessorFactory::create()
{
    return create("NullResultProcessor");
}

}

}



