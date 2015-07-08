/*  \file   ActivationCostFunctionFactory.cpp
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ActivationCostFunctionFactory class.
*/

// Lucious Includes
#include <lucious/network/interface/ActivationCostFunctionFactory.h>
#include <lucious/network/interface/NullActivationCostFunction.h>

namespace lucious
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


