/*  \file   CpuTargetOperationFactory.cpp
    \author Gregory Diamos
    \date   August 16, 2017
    \brief  The source file for the CpuTargetOperationFactory class.
*/

// Lucius Includes
#include <lucius/machine/cpu/interface/CpuTargetOperationFactory.h>

#include <lucius/machine/generic/interface/CallOperation.h>
#include <lucius/machine/generic/interface/ReturnOperation.h>
#include <lucius/machine/generic/interface/BranchOperation.h>
#include <lucius/machine/generic/interface/PHIOperation.h>

#include <lucius/machine/generic/interface/TensorValue.h>
#include <lucius/machine/generic/interface/IntegerValue.h>

#include <lucius/machine/cpu/interface/BinaryApplyOperation.h>
#include <lucius/machine/cpu/interface/ZerosOperation.h>
#include <lucius/machine/cpu/interface/CopyOperation.h>
#include <lucius/machine/cpu/interface/LessOperation.h>

#include <lucius/ir/interface/Type.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/target/interface/TargetOperation.h>
#include <lucius/ir/target/interface/TargetValue.h>

namespace lucius
{

namespace machine
{

namespace cpu
{

CpuTargetOperationFactory::CpuTargetOperationFactory(Context& context)
: TargetOperationFactory(context)
{

}

CpuTargetOperationFactory::~CpuTargetOperationFactory()
{
    // intentionally blank
}

ir::TargetOperation CpuTargetOperationFactory::create(const std::string& name) const
{
    if(name == "call")
    {
        return generic::CallOperation();
    }
    else if(name == "return")
    {
        return generic::ReturnOperation();
    }
    else if(name == "branch")
    {
        return generic::BranchOperation();
    }
    else if(name == "phi")
    {
        return generic::PHIOperation();
    }
    else if(name == "cpu-binary-apply")
    {
        return BinaryApplyOperation();
    }
    else if(name == "cpu-zeros")
    {
        return ZerosOperation();
    }
    else if(name == "cpu-copy")
    {
        return CopyOperation();
    }
    else if(name == "cpu-less")
    {
        return LessOperation();
    }

    throw std::runtime_error("No support for creating operation named '" + name +
        "' for CPU target machine.");
}

ir::TargetValue CpuTargetOperationFactory::createOperand(const ir::Type& t) const
{
    if(t.isTensor())
    {
        auto tensorType = ir::type_cast<ir::TensorType>(t);

        return generic::TensorValue(tensorType.getShape(), tensorType.getPrecision(), _context);
    }
    if(t.isInteger())
    {
        return generic::IntegerValue(_context);
    }

    throw std::runtime_error("No support for creating operand of type '" + t.toString() +
        "' for CPU target machine.");
}

} // namespace cpu
} // namespace machine
} // namespace lucius





