/*  \file   LayerFactory.h
    \date   November 12, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the LayerFactory class
*/

#pragma once

// Minerva Includes
#include <minerva/util/interface/ParameterPack.h>

// Standard Library
#include <string>
#include <memory>

// Forward Declarations
namespace minerva { namespace network { class Layer; } }

namespace minerva
{

namespace network
{

/* \brief A neural network layer factory. */
class LayerFactory
{
public:
    using ParameterPack = util::ParameterPack;

public:
    static std::unique_ptr<Layer> create(const std::string& name);

public:
    static std::unique_ptr<Layer> create(const std::string& name, const ParameterPack& );

public:
    template<typename... Args>
    static std::unique_ptr<Layer> create(const std::string& name, Args... args)
    {
        return create(name, ParameterPack(args...));
    }
};

}

}

