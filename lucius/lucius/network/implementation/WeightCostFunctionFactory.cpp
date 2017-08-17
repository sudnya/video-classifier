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

std::unique_ptr<WeightCostFunction> WeightCostFunctionFactory::create(
    const std::string& costFunctionName)
{
    if(costFunctionName == "WeightRegularizationCostFunction")
    {
        return std::make_unique<WeightRegularizationCostFunction>();
    }
    else if(costFunctionName == "NullWeightRegularizationCostFunction")
    {
        return std::make_unique<NullWeightRegularizationCostFunction>();
    }

    return std::unique_ptr<WeightCostFunction>();
}

std::unique_ptr<WeightCostFunction> WeightCostFunctionFactory::create()
{
    return create("NullWeightRegularizationCostFunction");
}

}

}


