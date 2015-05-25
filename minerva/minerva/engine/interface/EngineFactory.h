/*	\file   EngineFactory.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the EngineFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva { namespace engine { class Engine; } }

namespace minerva
{

namespace engine
{

/*! \brief A factory for classifier engines */
class EngineFactory
{
public:
	static Engine* create(const std::string& classifierName);

};

}

}


