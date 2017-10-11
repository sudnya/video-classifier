/*  \file   TargetOperation.cpp
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the TargetOperation class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/TargetOperation.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/PerformanceMetrics.h>

#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

namespace lucius
{

namespace ir
{

TargetOperation::TargetOperation()
{

}

TargetOperation::TargetOperation(std::shared_ptr<ValueImplementation> value)
: TargetOperation(std::static_pointer_cast<TargetOperationImplementation>(value))
{

}

TargetOperation::TargetOperation(std::shared_ptr<TargetOperationImplementation> value)
: _implementation(value)
{

}

TargetOperation::~TargetOperation()
{
    // intentionally blank
}

const Type& TargetOperation::getOutputType() const
{
    return getOutputOperand().getValue().getType();
}

PerformanceMetrics TargetOperation::getPerformanceMetrics() const
{
    return getTargetOperationImplementation()->getPerformanceMetrics();
}

void TargetOperation::execute()
{
    getTargetOperationImplementation()->execute();
}

void TargetOperation::setOutputOperand(const TargetValue& v)
{
    getTargetOperationImplementation()->setOutputOperand(v);
}

Use& TargetOperation::getOutputOperand()
{
    return getTargetOperationImplementation()->getOutputOperand();
}

const Use& TargetOperation::getOutputOperand() const
{
    return getTargetOperationImplementation()->getOutputOperand();
}

TargetOperation::UseList& TargetOperation::getAllOperands()
{
    return getTargetOperationImplementation()->getAllOperands();
}

const TargetOperation::UseList& TargetOperation::getAllOperands() const
{
    return getTargetOperationImplementation()->getAllOperands();
}

void TargetOperation::setOperand(const TargetValue& v, size_t index)
{
    getTargetOperationImplementation()->setOperand(v, index);
}

void TargetOperation::appendOperand(const TargetValue& v)
{
    getTargetOperationImplementation()->appendOperand(v);
}

std::shared_ptr<TargetOperationImplementation>
TargetOperation::getTargetOperationImplementation() const
{
    return std::static_pointer_cast<TargetOperationImplementation>(_implementation);
}

} // namespace ir
} // namespace lucius

