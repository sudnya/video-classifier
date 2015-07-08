/*	\file   ModelBuilder.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ModelBuilder class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucious { namespace model { class Model; } }

namespace lucious
{

namespace model
{

/*! \brief A constructor for classification models */
class ModelBuilder
{
public:
	/*! \brief Create a new model at the specified path */
	Model* create(const std::string& path);

public:
	/*! \brief Create a new model at the specified path using the
		specified topology. */
	Model* create(const std::string& path, const std::string& specificationPath);

};

}

}


