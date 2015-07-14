/*  \file   EngineObserverFactory.cpp
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the EngineObserverFactory class.
*/

// Lucious Includes
#include <lucious/engine/interface/EngineObserverFactory.h>
#include <lucious/engine/interface/EngineObserver.h>

#include <lucious/engine/interface/ModelCheckpointer.h>

// Standard Library Includes
#include <lucious/util/interface/memory.h>

namespace lucious
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

    return nullptr;
}

}

}




