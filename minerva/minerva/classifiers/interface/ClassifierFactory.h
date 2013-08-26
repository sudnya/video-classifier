/*	\file   ClassifierFactory.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ClassifierFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva { namespace classifiers { class ClassifierEngine; } }

namespace minerva
{

namespace classifiers
{

/*! \brief A factory for classifier engines */
class ClassifierFactory
{
public:
	static ClassifierEngine* create(const std::string& classifierName);

};

}

}


