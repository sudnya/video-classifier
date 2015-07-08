/*	\file   BuiltInSpecifications.h
	\date   Saturday April 26, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the BuiltInSpecifications class.
*/

#pragma once

// Standard Library Includes
#include <string>

namespace lucious
{

namespace model
{

/*! \brief A singleton used to provide built-in model specifications. */
class BuiltInSpecifications
{
public:
	static std::string getConvolutionalFastModelSpecification(size_t outputCount);

};

}

}

