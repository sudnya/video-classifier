/*  \file   CostFunctionFactory.h
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the CostFunctionFactory class.
*/

// Lucius Includes
#include <lucius/network/interface/CostFunctionFactory.h>

#include <lucius/network/interface/CTCCostFunction.h>
#include <lucius/network/interface/SumOfSquaresCostFunction.h>
#include <lucius/network/interface/SoftmaxCostFunction.h>

namespace lucius
{

namespace network
{

CostFunction* CostFunctionFactory::create(const std::string& name)
{
    if("SumOfSquaresCostFunction" == name)
    {
        return new SumOfSquaresCostFunction;
    }
    else if("SoftmaxCostFunction" == name)
    {
        return new SoftmaxCostFunction;
    }
    else if("CTCCostFunction" == name)
    {
        return new CTCCostFunction;
    }

    return nullptr;
}

CostFunction* CostFunctionFactory::create()
{
    return create("SumOfSquaresCostFunction");
}

}

}



