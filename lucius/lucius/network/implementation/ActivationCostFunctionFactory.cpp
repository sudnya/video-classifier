/*  \file   ActivationCostFunctionFactory.cpp
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ActivationCostFunctionFactory class.
*/

// Lucius Includes
#include <lucius/network/interface/ActivationCostFunctionFactory.h>
#include <lucius/network/interface/NullActivationCostFunction.h>

namespace lucius
{

namespace network
{

ActivationCostFunction* ActivationCostFunctionFactory::create(const std::string& name)
{
    if(name == "NullActivationCostFunction")
    {
        return new NullActivationCostFunction;
    }

    return nullptr;
}

ActivationCostFunction* ActivationCostFunctionFactory::create()
{
    return create("KLDivergenceActivationCostFunction");
}

}

}


