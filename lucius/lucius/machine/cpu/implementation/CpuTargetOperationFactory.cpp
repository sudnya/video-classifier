/*  \file   CpuTargetOperationFactory.cpp
    \author Gregory Diamos
    \date   August 16, 2017
    \brief  The source file for the CpuTargetOperationFactory class.
*/

// Lucius Includes
#include <lucius/machine/cpu/interface/CpuTargetOperationFactory.h>

#include <lucius/machine/generic/interface/CallOperation.h>
#include <lucius/machine/generic/interface/ReturnOperation.h>

#include <lucius/machine/cpu/interface/BinaryApplyOperation.h>

#include <lucius/ir/target/interface/TargetOperation.h>

namespace lucius
{

namespace machine
{

namespace cpu
{

CpuTargetOperationFactory::~CpuTargetOperationFactory()
{
    // intentionally blank
}

ir::TargetOperation CpuTargetOperationFactory::create(const std::string& name)
{
    if(name == "call")
    {
        return generic::CallOperation();
    }
    else if(name == "return")
    {
        return generic::ReturnOperation();
    }
    else if(name == "cpu-binary-apply")
    {
        return BinaryApplyOperation();
    }

    throw std::runtime_error("No support for creating operation named '" + name +
        "' for CPU target machine.");
}

} // namespace cpu
} // namespace machine
} // namespace lucius





