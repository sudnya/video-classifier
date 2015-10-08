/*  \file   WeihtCostFunctionFactory.cpp
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the WeightCostFunctionFactory class.
*/

// Lucius Includes
#include <lucius/network/interface/WeightCostFunctionFactory.h>

#include <lucius/network/interface/WeightRegularizationCostFunction.h>
#include <lucius/network/interface/NullWeightRegularizationCostFunction.h>

namespace lucius
{

namespace network
{

WeightCostFunction* WeightCostFunctionFactory::create(const std::string& costFunctionName)
{
    if(costFunctionName == "WeightRegularizationCostFunction")
    {
        return new WeightRegularizationCostFunction;
    }
    else if(costFunctionName == "NullWeightRegularizationCostFunction")
    {
        return new NullWeightRegularizationCostFunction;
    }

    return nullptr;
}

WeightCostFunction* WeightCostFunctionFactory::create()
{
    return create("NullWeightRegularizationCostFunction");
}

}

}


