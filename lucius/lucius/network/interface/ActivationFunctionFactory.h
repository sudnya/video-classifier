/*    \file   ActivationFunctionFactory.h
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ActivationFunctionFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucius { namespace network { class ActivationFunction; } }

namespace lucius
{

namespace network
{

/*! \brief A factory for cost functions. */
class ActivationFunctionFactory
{
public:
    static std::unique_ptr<ActivationFunction> create(const std::string& functionName);
    static std::unique_ptr<ActivationFunction> create();

};

}

}




