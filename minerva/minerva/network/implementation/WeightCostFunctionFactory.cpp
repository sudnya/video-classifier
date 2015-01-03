/*	\file   WeihtCostFunctionFactory.cpp
	\date   December 25, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the WeightCostFunctionFactory class.
*/

// Minerva Includes
#include <minerva/network/interface/WeightCostFunctionFactory.h>

namespace minerva
{

namespace network
{

WeightCostFunction* WeightCostFunctionFactory::create(const std::string& costFunctionName)
{
	return nullptr;
}

WeightCostFunction* WeightCostFunctionFactory::create()
{
	return create("WeightRegularizationCostFunction");
}

}

}


