/*  \file   LayerFactory.h
    \date   November 12, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the LayerFactory class
*/

#pragma once

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
    static std::unique_ptr<Layer> create(const std::string& name);
};

}

}

