/*  \file   WeihtCostFunctionFactory.cpp
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the WeightCostFunctionFactory class.
*/

// Lucious Includes
#include <lucious/network/interface/WeightCostFunctionFactory.h>

#include <lucious/network/interface/WeightRegularizationCostFunction.h>

namespace lucious
{

namespace network
{

WeightCostFunction* WeightCostFunctionFactory::create(const std::string& costFunctionName)
{
    if(costFunctionName == "WeightRegularizationCostFunction")
    {
        return new WeightRegularizationCostFunction;
    }

    return nullptr;
}

WeightCostFunction* WeightCostFunctionFactory::create()
{
    return create("WeightRegularizationCostFunction");
}

}

}


