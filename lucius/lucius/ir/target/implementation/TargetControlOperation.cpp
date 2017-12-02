/*  \file   TargetControlOperation.h
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the TargetControlOperation class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/TargetControlOperation.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Operation.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>

#include <lucius/ir/target/implementation/TargetControlOperationImplementation.h>

namespace lucius
{

namespace ir
{

TargetControlOperation::TargetControlOperation(
    std::shared_ptr<TargetOperationImplementation> implementation)
: TargetOperation(implementation)
{

}

TargetControlOperation::TargetControlOperation(Operation op)
: TargetOperation(op.getValueImplementation())
{

}

TargetControlOperation::~TargetControlOperation()
{
    // intentionally blank
}

PerformanceMetrics TargetControlOperation::getPerformanceMetrics() const
{
    return getTargetOperationImplementation()->getPerformanceMetrics();
}

BasicBlock TargetControlOperation::execute()
{
    return getTargetControlOperationImplementation()->execute();
}

std::shared_ptr<TargetControlOperationImplementation>
    TargetControlOperation::getTargetControlOperationImplementation() const
{
    return std::static_pointer_cast<TargetControlOperationImplementation>(
        getValueImplementation());
}

} // namespace ir
} // namespace lucius



