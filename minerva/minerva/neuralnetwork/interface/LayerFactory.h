/*  \file   LayerFactory.h
    \date   November 12, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the LayerFactory class
*/

#pragma once

// Forward Declarations
namespace minerva { namespace neuralnetwork { class Layer; } }

namespace minerva
{
namespace neuralnetwork
{

/* \brief A neural network layer factory. */
class LayerFactory
{
public:
    static Layer* create(const std::string& name);
};

}
}

