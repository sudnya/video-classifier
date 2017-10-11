/*  \file   TargetControlOperation.h
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the TargetControlOperation class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/TargetControlOperation.h>

#include <lucius/ir/interface/BasicBlock.h>

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

} // namespace ir
} // namespace lucius



