/*    \file   CostFunctionFactory.h
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the CostFunctionFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <memory>

// Forward Declarations
namespace lucius { namespace network { class CostFunction; } }

namespace lucius
{

namespace network
{

/*! \brief A factory for cost functions. */
class CostFunctionFactory
{
public:
    static std::unique_ptr<CostFunction> create(const std::string& costFunctionName);

    // Create the default cost function
    static std::unique_ptr<CostFunction> create();

};

}

}


