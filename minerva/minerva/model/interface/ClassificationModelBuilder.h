/*	\file   ClassificationModelBuilder.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ClassificationModelBuilder class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva { namespace model { class ClassificationModel; } }

namespace minerva
{

namespace model
{

/*! \brief A constructor for classification models */
class ClassificationModelBuilder
{
public:
	ClassificationModel* create(const std::string& path);

};

}

}


