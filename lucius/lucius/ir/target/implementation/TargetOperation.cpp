/*  \file   TargetOperation.cpp
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the TargetOperation class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/TargetOperation.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/User.h>
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
    _implementation->setImplementation(_implementation);
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

TargetOperation::iterator TargetOperation::getOutputOperandPosition()
{
    return getTargetOperationImplementation()->getOutputOperandPosition();
}

TargetOperation::const_iterator TargetOperation::getOutputOperandPosition() const
{
    return getTargetOperationImplementation()->getOutputOperandPosition();
}

bool TargetOperation::hasOutputOperand() const
{
    return getTargetOperationImplementation()->hasOutputOperand();
}

Use& TargetOperation::getOperand(size_t index)
{
    assert(index < getOperands().size());

    auto operand = getOperands().begin();

    std::advance(operand, index);

    return *operand;
}

const Use& TargetOperation::getOperand(size_t index) const
{
    assert(index < getOperands().size());

    auto operand = getOperands().begin();

    std::advance(operand, index);

    return *operand;
}

TargetOperation::iterator TargetOperation::getOperandPosition(size_t index)
{
    assert(index < getOperands().size());

    auto operand = getOperands().begin();

    std::advance(operand, index);

    return operand;
}

TargetOperation::const_iterator TargetOperation::getOperandPosition(size_t index) const
{
    assert(index < getOperands().size());

    auto operand = getOperands().begin();

    std::advance(operand, index);

    return operand;
}

TargetOperation::iterator TargetOperation::begin()
{
    return getOperands().begin();
}

TargetOperation::const_iterator TargetOperation::begin() const
{
    return getOperands().begin();
}

TargetOperation::iterator TargetOperation::end()
{
    return getOperands().end();
}

TargetOperation::const_iterator TargetOperation::end() const
{
    return getOperands().end();
}

TargetOperation::IteratorRange TargetOperation::getInputOperandRange()
{
    return IteratorRange(begin(), hasOutputOperand() ? --end() : end());
}

TargetOperation::ConstIteratorRange TargetOperation::getInputOperandRange() const
{
    return ConstIteratorRange(begin(), hasOutputOperand() ? --end() : end());
}

TargetOperation::UseList& TargetOperation::getOperands()
{
    return getTargetOperationImplementation()->getOperands();
}

const TargetOperation::UseList& TargetOperation::getOperands() const
{
    return getTargetOperationImplementation()->getOperands();
}

bool TargetOperation::hasInputOperands() const
{
    return getInputOperandCount() != 0;
}

void TargetOperation::setOperand(const TargetValue& v, size_t index)
{
    getTargetOperationImplementation()->setOperand(v, index);
}

void TargetOperation::appendOperand(const TargetValue& v)
{
    getTargetOperationImplementation()->appendOperand(v);
}

void TargetOperation::replaceOperand(const Use& original, const Use& newOperand)
{
    getTargetOperationImplementation()->replaceOperand(original, newOperand);
}

size_t TargetOperation::getInputOperandCount() const
{
    if(hasOutputOperand())
    {
        return getOperands().size() - 1;
    }

    return getOperands().size();
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

bool TargetOperation::isPHI() const
{
    return _implementation->isPHI();
}

BasicBlock TargetOperation::getBasicBlock() const
{
    return _implementation->getBasicBlock();
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

