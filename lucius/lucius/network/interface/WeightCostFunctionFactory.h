/*    \file   WeihtCostFunctionFactory.h
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the WeightCostFunctionFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucius { namespace network { class WeightCostFunction; } }

namespace lucius
{

namespace network
{

/*! \brief A factory for weight cost functions. */
class WeightCostFunctionFactory
{
public:
    static std::unique_ptr<WeightCostFunction> create(const std::string& costFunctionName);
    static std::unique_ptr<WeightCostFunction> create();

};

}

}

