/*    \file   ActivationFunctionFactory.h
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ActivationFunctionFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva { namespace network { class ActivationFunction; } }

namespace minerva
{

namespace network
{

/*! \brief A factory for cost functions. */
class ActivationFunctionFactory
{
public:
    static ActivationFunction* create(const std::string& functionName);
    static ActivationFunction* create();

};

}

}




