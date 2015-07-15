/*    \file   ActivationFunctionFactory.cpp
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ActivationFunctionFactory class.
*/

// Lucius Includes
#include <lucius/network/interface/ActivationFunctionFactory.h>

#include <lucius/network/interface/SigmoidActivationFunction.h>
#include <lucius/network/interface/RectifiedLinearActivationFunction.h>
#include <lucius/network/interface/NullActivationFunction.h>

namespace lucius
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


