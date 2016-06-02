/*  \file   EngineFactory.h
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the EngineFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <memory>

// Forward Declarations
namespace lucius { namespace engine { class Engine; } }

namespace lucius
{

namespace engine
{

/*! \brief A factory for classifier engines */
class EngineFactory
{
public:
    static std::unique_ptr<Engine> create(const std::string& classifierName);

};

}

}


