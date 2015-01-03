/*	\file   ActivationCostFunctionFactory.cpp
	\date   December 25, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ActivationCostFunctionFactory class.
*/

// Minerva Includes
#include <minerva/network/interface/ActivationCostFunctionFactory.h>

namespace minerva
{

namespace network
{

ActivationCostFunction* ActivationCostFunctionFactory::create(const std::string& name)
{
	return nullptr;
}

ActivationCostFunction* ActivationCostFunctionFactory::create()
{
	return create("KLDivergenceActivationCostFunction");
}

}

}


