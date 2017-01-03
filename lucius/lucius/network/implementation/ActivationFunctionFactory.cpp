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

#include <lucius/util/interface/memory.h>

namespace lucius
{

namespace network
{

std::unique_ptr<ActivationFunction> ActivationFunctionFactory::create(const std::string& name)
{
    if(name == "RectifiedLinearActivationFunction")
    {
        return std::make_unique<RectifiedLinearActivationFunction>();
    }
    else if(name == "SigmoidActivationFunction")
    {
        return std::make_unique<SigmoidActivationFunction>();
    }
    else if(name == "NullActivationFunction")
    {
        return std::make_unique<NullActivationFunction>();
    }

    return std::unique_ptr<ActivationFunction>();
}

std::unique_ptr<ActivationFunction> ActivationFunctionFactory::create()
{
    return create("RectifiedLinearActivationFunction");
}

}

}


