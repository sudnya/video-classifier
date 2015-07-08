/*    \file   ActivationFunctionFactory.cpp
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ActivationFunctionFactory class.
*/

// Lucious Includes
#include <lucious/network/interface/ActivationFunctionFactory.h>

#include <lucious/network/interface/SigmoidActivationFunction.h>
#include <lucious/network/interface/RectifiedLinearActivationFunction.h>
#include <lucious/network/interface/NullActivationFunction.h>

namespace lucious
{

namespace network
{

ActivationFunction* ActivationFunctionFactory::create(const std::string& name)
{
    if(name == "RectifiedLinearActivationFunction")
    {
        return new RectifiedLinearActivationFunction;
    }
    else if (name == "SigmoidActivationFunction")
    {
        return new SigmoidActivationFunction;
    }
    else if (name == "NullActivationFunction")
    {
        return new NullActivationFunction;
    }

    return nullptr;
}

ActivationFunction* ActivationFunctionFactory::create()
{
    return create("RectifiedLinearActivationFunction");
}

}

}


