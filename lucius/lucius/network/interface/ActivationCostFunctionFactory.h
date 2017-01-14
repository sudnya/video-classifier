/*  \file   ActivationCostFunctionFactory.h
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ActivationCostFunctionFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <memory>

// Forward Declarations
namespace lucius { namespace network { class ActivationCostFunction; } }

namespace lucius
{

namespace network
{

/*! \brief A factory for cost functions. */
class ActivationCostFunctionFactory
{
public:
    static std::unique_ptr<ActivationCostFunction> create(const std::string& costFunctionName);
    static std::unique_ptr<ActivationCostFunction> create();

};

}

}



