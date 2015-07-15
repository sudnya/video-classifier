/*  \file   ActivationCostFunctionFactory.h
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ActivationCostFunctionFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucious { namespace network { class ActivationCostFunction; } }

namespace lucious
{

namespace network
{

/*! \brief A factory for cost functions. */
class ActivationCostFunctionFactory
{
public:
    static ActivationCostFunction* create(const std::string& costFunctionName);
    static ActivationCostFunction* create();

};

}

}



