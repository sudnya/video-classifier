/*    \file   CostFunctionFactory.h
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the CostFunctionFactory class.
*/

// Minerva Includes
#include <minerva/network/interface/CostFunctionFactory.h>

#include <minerva/network/interface/SumOfSquaresCostFunction.h>
#include <minerva/network/interface/SoftMaxCostFunction.h>

namespace minerva
{

namespace network
{

CostFunction* CostFunctionFactory::create(const std::string& name)
{
    if("SumOfSquaresCostFunction" == name)
    {
        return new SumOfSquaresCostFunction;
    }
    else if("SoftMaxCostFunction" == name)
    {
        return new SoftmaxCostFunction;
    }

    return nullptr;
}

CostFunction* CostFunctionFactory::create()
{
    return create("SumOfSquaresCostFunction");
}

}

}



