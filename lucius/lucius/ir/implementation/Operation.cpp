/*  \file   Operation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Operation class.
*/

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

#include <lucius/ir/interface/BasicBlock.h>

#include <lucius/ir/target/interface/TargetOperation.h>

#include <lucius/ir/ops/implementation/ControlOperationImplementation.h>
#include <lucius/ir/ops/implementation/ComputeGradientOperationImplementation.h>
#include <lucius/ir/ops/implementation/PHIOperationImplementation.h>

#include <lucius/ir/target/implementation/TargetControlOperationImplementation.h>

#include <lucius/ir/implementation/OperationImplementation.h>

#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace ir
{

using UseList = std::list<Use>;
using OperationList = std::list<Operation>;
using ValueList = std::list<Value>;

Operation::Operation()
{

}

Operation::Operation(std::shared_ptr<ValueImplementation> implementation)
: _implementation(std::static_pointer_cast<OperationImplementation>(implementation))
{
    assert(Value(implementation).isOperation());
}

Operation::Operation(const TargetOperation& op)
: Operation(op.getValueImplementation())
{

}

Operation::~Operation()
{
    // intentionally blank
}

ShapeList Operation::getOutputShapes(const ShapeList& inputShapes) const
{
    return _implementation->getOutputShapes(inputShapes);
}

ShapeList Operation::getInputShapes(const ShapeList& outputShapes) const
{
    return _implementation->getInputShapes(outputShapes);
}

Operation::iterator Operation::begin()
{
    return getOperands().begin();
}

Operation::const_iterator Operation::begin() const
{
    return getOperands().begin();
}

Operation::iterator Operation::end()
{
    return getOperands().end();
}

Operation::const_iterator Operation::end() const
{
    return getOperands().end();
}

const UseList& Operation::getOperands() const
{
    return _implementation->getOperands();
}

UseList& Operation::getOperands()
{
    return _implementation->getOperands();
}

size_t Operation::size() const
{
    return _implementation->size();
}

bool Operation::empty() const
{
    return _implementation->empty();
}

const Use& Operation::getOperand(size_t index) const
{
    return _implementation->getOperand(index);
}

Use& Operation::getOperand(size_t index)
{
    return _implementation->getOperand(index);
}

void Operation::setOperands(const UseList& uses)
{
    _implementation->setOperands(uses);

    for(auto position = begin(); position != end(); ++position)
    {
        auto& use = *position;

        use.setParent(User(_implementation), position);
        use.getValue().addUse(use);

        use.getValue().bindToContextIfDifferent(getContext());
    }
}

void Operation::setOperands(const ValueList& values)
{
    UseList list;

    for(auto& value : values)
    {
        list.push_back(Use(value));
    }

    setOperands(list);
}

void Operation::appendOperand(const Use& use)
{
    insertOperand(end(), use);
}

void Operation::appendOperand(const Value& value)
{
    appendOperand(Use(value));
}

void Operation::replaceOperand(const Use& original, const Use& newOperand)
{
    auto position = begin();

    for( ; position != end(); ++position)
    {
        if(position->getValue() == original.getValue())
        {
            auto& operand = *position;
            ++position;

            operand.detach();
            break;
        }
    }

    insertOperand(position, newOperand);
}

void Operation::insertOperand(iterator position, const Use& operand)
{
    auto newPosition = getOperands().insert(position, operand);

    newPosition->setParent(User(_implementation), newPosition);
    newPosition->getValue().addUse(*newPosition);

    newPosition->getValue().bindToContextIfDifferent(getContext());
}

OperationList Operation::getPredecessors() const
{
    OperationList operations;

    for(auto& use : getOperands())
    {
        if(!use.getValue().isOperation())
        {
            continue;
        }

        operations.push_back(Operation(use.getOperation()));
    }

    return operations;
}

OperationList Operation::getSuccessors() const
{
    OperationList operations;

    for(auto& use : _implementation->getUses())
    {
        operations.push_back(use.getOperation());
    }

    return operations;
}

ValueList Operation::getUsedValues() const
{
    ValueList values;

    for(auto& use : getOperands())
    {
        values.push_back(use.getValue());
    }

    return values;
}

bool Operation::isControlOperation() const
{
    return static_cast<bool>(std::dynamic_pointer_cast<ControlOperationImplementation>(
            getValueImplementation())) ||
        static_cast<bool>(std::dynamic_pointer_cast<TargetControlOperationImplementation>(
            getValueImplementation()));
}

bool Operation::isGradientOperation() const
{
    return static_cast<bool>(std::dynamic_pointer_cast<ComputeGradientOperationImplementation>(
        getValueImplementation()));
}

bool Operation::isPHI() const
{
    return static_cast<bool>(std::dynamic_pointer_cast<PHIOperationImplementation>(
        getValueImplementation()));
}

bool Operation::isReturn() const
{
    return _implementation->isReturn();
}

bool Operation::isCall() const
{
    return _implementation->isCall();
}

Type Operation::getType() const
{
    return _implementation->getType();
}

BasicBlock Operation::getBasicBlock() const
{
    return getParent();
}

BasicBlock Operation::getParent() const
{
    return _implementation->getParent();
}

void Operation::setParent(const BasicBlock& parent)
{
    _implementation->setParent(parent);
}

void Operation::setIterator(Operation::operation_iterator iterator)
{
    _implementation->setIterator(iterator);
}

Operation::operation_iterator Operation::getIterator()
{
    return _implementation->getIterator();
}

Operation::const_operation_iterator Operation::getIterator() const
{
    return _implementation->getIterator();
}

Context* Operation::getContext()
{
    return _implementation->getContext();
}

Operation Operation::clone() const
{
    return Operation(getValueImplementation()->clone());
}

std::string Operation::name() const
{
    return _implementation->name();
}

std::string Operation::toString() const
{
    return _implementation->toString();
}

std::shared_ptr<ValueImplementation> Operation::getValueImplementation() const
{
    return _implementation;
}

std::shared_ptr<OperationImplementation> Operation::getImplementation() const
{
    return _implementation;
}

bool Operation::operator==(const Operation& operation) const
{
    return getValueImplementation()->getId() == operation.getValueImplementation()->getId();
}

bool Operation::operator<(const Operation& operation) const
{
    return getValueImplementation()->getId() < operation.getValueImplementation()->getId();
}

} // namespace ir
} // namespace lucius

