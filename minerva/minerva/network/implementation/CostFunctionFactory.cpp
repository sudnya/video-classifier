/*    \file   CostFunctionFactory.h
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the CostFunctionFactory class.
*/

// Lucious Includes
#include <lucious/network/interface/CostFunctionFactory.h>

#include <lucious/network/interface/SumOfSquaresCostFunction.h>
#include <lucious/network/interface/SoftmaxCostFunction.h>

namespace lucious
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



