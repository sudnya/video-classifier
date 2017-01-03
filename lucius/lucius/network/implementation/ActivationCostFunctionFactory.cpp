/*  \file   ActivationCostFunctionFactory.cpp
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ActivationCostFunctionFactory class.
*/

// Lucius Includes
#include <lucius/network/interface/ActivationCostFunctionFactory.h>
#include <lucius/network/interface/NullActivationCostFunction.h>

#include <lucius/util/interface/memory.h>

namespace lucius
{

namespace network
{

std::unique_ptr<ActivationCostFunction> ActivationCostFunctionFactory::create(
    const std::string& name)
{
    if(name == "NullActivationCostFunction")
    {
        return std::make_unique<NullActivationCostFunction>();
    }

    return std::unique_ptr<ActivationCostFunction>();
}

std::unique_ptr<ActivationCostFunction> ActivationCostFunctionFactory::create()
{
    return create("KLDivergenceActivationCostFunction");
}

}

}


