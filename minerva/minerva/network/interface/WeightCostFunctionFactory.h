/*    \file   WeihtCostFunctionFactory.h
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the WeightCostFunctionFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva { namespace network { class WeightCostFunction; } }

namespace minerva
{

namespace network
{

/*! \brief A factory for weight cost functions. */
class WeightCostFunctionFactory
{
public:
    static WeightCostFunction* create(const std::string& costFunctionName);
    static WeightCostFunction* create();

};

}

}

