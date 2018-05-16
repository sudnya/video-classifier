/*  \file   PHIOperation.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the PHIOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/PHIOperation.h>

#include <lucius/ir/ops/implementation/PHIOperationImplementation.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace ir
{

PHIOperation::PHIOperation(const Type& type)
: PHIOperation(std::make_shared<PHIOperationImplementation>(type))
{

}

PHIOperation::PHIOperation(std::shared_ptr<ValueImplementation> implementation)
: Operation(implementation)
{

}

PHIOperation::~PHIOperation()
{

}

const PHIOperation::BasicBlockVector& PHIOperation::getIncomingBasicBlocks() const
{
    return std::static_pointer_cast<PHIOperationImplementation>(
        getImplementation())->getIncomingBasicBlocks();
}

void PHIOperation::addIncomingValue(const Value& value, const BasicBlock& predecessor)
{
    std::static_pointer_cast<PHIOperationImplementation>(
        getImplementation())->addIncomingValue(value, predecessor);
}

} // namespace ir
} // namespace lucius






