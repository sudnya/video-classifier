/*! \file   BackPropagationFactory.h
	\author Gregory Diamos
	\date   Sunday December 22, 2013
	\brief  The header file for the BackPropagationFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva { namespace neuralnetwork { class BackPropagation; } }

namespace minerva
{

namespace neuralnetwork
{

class BackPropagationFactory
{
public:
	static BackPropagation* create(const std::string& name);

};

}

}
