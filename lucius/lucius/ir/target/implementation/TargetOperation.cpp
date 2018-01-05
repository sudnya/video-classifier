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

// Standard Library Includes
#include <string>
#include <cassert>

namespace lucius
{

namespace ir
{

TargetOperation::TargetOperation()
{

}

TargetOperation::TargetOperation(std::shared_ptr<ValueImplementation> value)
: _implementation(std::static_pointer_cast<TargetOperationImplementation>(value))
{
    assert(Value(value).isTargetOperation());
}

TargetOperation::~TargetOperation()
{
    // intentionally blank
}

Type TargetOperation::getOutputType() const
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
    TargetValue(v).addDefinition(Use(*this));

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

Use& TargetOperation::getOperand(size_t index)
{
    auto operand = getOperands().begin();

    std::advance(operand, index);

    return *operand;
}

const Use& TargetOperation::getOperand(size_t index) const
{
    auto operand = getOperands().begin();

    std::advance(operand, index);

    return *operand;
}

TargetOperation::UseList& TargetOperation::getOperands()
{
    return getTargetOperationImplementation()->getOperands();
}

const TargetOperation::UseList& TargetOperation::getOperands() const
{
    return getTargetOperationImplementation()->getOperands();
}

void TargetOperation::setOperand(const TargetValue& v, size_t index)
{
    getTargetOperationImplementation()->setOperand(v, index);
}

void TargetOperation::appendOperand(const TargetValue& v)
{
    getTargetOperationImplementation()->appendOperand(v);
}

bool TargetOperation::isValid() const
{
    return static_cast<bool>(_implementation);
}

bool TargetOperation::isCall() const
{
    return _implementation->isCall();
}

bool TargetOperation::isReturn() const
{
    return _implementation->isReturn();
}

std::string TargetOperation::toString() const
{
    return _implementation->toString();
}

std::shared_ptr<TargetOperationImplementation>
TargetOperation::getTargetOperationImplementation() const
{
    return std::static_pointer_cast<TargetOperationImplementation>(_implementation);
}

std::shared_ptr<ValueImplementation> TargetOperation::getValueImplementation() const
{
    return _implementation;
}

} // namespace ir
} // namespace lucius

