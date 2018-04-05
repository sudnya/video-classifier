/*  \file   ControlOperation.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the ControlOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/ControlOperation.h>

#include <lucius/ir/interface/BasicBlock.h>

#include <lucius/ir/ops/implementation/ControlOperationImplementation.h>
#include <lucius/ir/target/implementation/TargetControlOperationImplementation.h>

namespace lucius
{

namespace ir
{

ControlOperation::ControlOperation(std::shared_ptr<ValueImplementation> implementation)
: Operation(implementation)
{

}

ControlOperation::~ControlOperation()
{

}

ControlOperation::BasicBlockVector ControlOperation::getPossibleTargets() const
{
    auto normalControlOperation = std::dynamic_pointer_cast<ControlOperationImplementation>(
        getValueImplementation());

    if(normalControlOperation)
    {
        return normalControlOperation->getPossibleTargets();
    }

    auto targetControlOperation = std::dynamic_pointer_cast<TargetControlOperationImplementation>(
        getValueImplementation());

    return targetControlOperation->getPossibleTargets();
}

} // namespace ir
} // namespace lucius





