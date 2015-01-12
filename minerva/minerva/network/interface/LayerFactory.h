/*  \file   LayerFactory.h
    \date   November 12, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the LayerFactory class
*/

#pragma once

// Standard Library
#include <string>

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
    static Layer* create(const std::string& name);
};

}

}

