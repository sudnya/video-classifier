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

#include <lucius/util/interface/memory.h>

namespace lucius
{

namespace network
{

std::unique_ptr<CostFunction> CostFunctionFactory::create(const std::string& name)
{
    if("SumOfSquaresCostFunction" == name)
    {
        return std::make_unique<SumOfSquaresCostFunction>();
    }
    else if("SoftmaxCostFunction" == name)
    {
        return std::make_unique<SoftmaxCostFunction>();
    }
    else if("CTCCostFunction" == name)
    {
        return std::make_unique<CTCCostFunction>();
    }

    return std::unique_ptr<CostFunction>(nullptr);
}

std::unique_ptr<CostFunction> CostFunctionFactory::create()
{
    return create("SumOfSquaresCostFunction");
}

}

}



