/*  \file   EngineObserverFactory.h
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the EngineObserverFactory class.
*/

#pragma once

// Lucious Includes
#include <lucious/util/interface/ParameterPack.h>

// Standard Library Includes
#include <string>
#include <memory>

// Forward Declarations
namespace lucious { namespace engine { class EngineObserver; } }

namespace lucious
{

namespace engine
{

/*! \brief A factory for classifier engine observers */
class EngineObserverFactory
{
public:
    using ParameterPack = util::ParameterPack;

public:
    static std::unique_ptr<EngineObserver> create(const std::string& name);
    static std::unique_ptr<EngineObserver> create(const std::string& name,
        const ParameterPack& );

public:
    template<typename... Args>
    static std::unique_ptr<EngineObserver> create(const std::string& name, Args... args)
    {
        return create(name, ParameterPack(args...));
    }

};

}

}



