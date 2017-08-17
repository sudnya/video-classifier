/*  \file   EngineObserverFactory.cpp
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the EngineObserverFactory class.
*/

// Lucius Includes
#include <lucius/engine/interface/EngineObserverFactory.h>
#include <lucius/engine/interface/EngineObserver.h>

#include <lucius/engine/interface/ModelCheckpointer.h>
#include <lucius/engine/interface/ValidationErrorObserver.h>

namespace lucius
{

namespace engine
{

std::unique_ptr<EngineObserver> EngineObserverFactory::create(const std::string& name)
{
    return create(name, ParameterPack());
}

std::unique_ptr<EngineObserver> EngineObserverFactory::create(const std::string& name,
    const ParameterPack& parameters)
{
    if(name == "ModelCheckpointer")
    {
        auto path = parameters.get<std::string>("Path", "");

        return std::make_unique<ModelCheckpointer>(path);
    }
    else if(name == "ValidationErrorObserver")
    {
        auto inputPath  = parameters.get<std::string>("InputPath",  "");
        auto outputPath = parameters.get<std::string>("OutputPath", "");
        auto batchSize  = parameters.get<size_t>("BatchSize", 1);
        auto maximumSamples = parameters.get<size_t>("MaximumSamples", 10);

        return std::make_unique<ValidationErrorObserver>(inputPath, outputPath, batchSize,
            maximumSamples);
    }

    return nullptr;
}

}

}




