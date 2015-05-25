/*	\file   ActivationFunctionFactory.cpp
	\date   December 25, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ActivationFunctionFactory class.
*/

// Minerva Includes
#include <minerva/network/interface/ActivationFunctionFactory.h>

#include <minerva/network/interface/SigmoidActivationFunction.h>
#include <minerva/network/interface/RectifiedLinearActivationFunction.h>
#include <minerva/network/interface/NullActivationFunction.h>

namespace minerva
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


