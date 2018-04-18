/*  \file   Operation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Operation class.
*/

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

#include <lucius/ir/interface/BasicBlock.h>

#include <lucius/ir/target/interface/TargetOperation.h>
#include <lucius/ir/target/interface/TargetValue.h>

#include <lucius/ir/ops/implementation/ControlOperationImplementation.h>
#include <lucius/ir/ops/implementation/ComputeGradientOperationImplementation.h>

#include <lucius/ir/target/implementation/TargetControlOperationImplementation.h>

#include <lucius/ir/implementation/OperationImplementation.h>

#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>

#include <lucius/util/interface/debug.h>

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
    _implementation->setImplementation(_implementation);
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

}

void Operation::setOperands(const ValueList& values)
{
    _implementation->setOperands(values);
}

void Operation::appendOperand(const Use& use)
{
    _implementation->appendOperand(use);
}

void Operation::appendOperand(const Value& value)
{
    _implementation->appendOperand(value);
}

void Operation::replaceOperand(const Use& original, const Use& newOperand)
{
    _implementation->replaceOperand(original, newOperand);
}

void Operation::insertOperand(iterator position, const Use& operand)
{
    _implementation->insertOperand(position, operand);
}

OperationList Operation::getPredecessors() const
{
    OperationList operations;

    // Target operations need to be handled differently because they represent defined
    // values explicitly as operands
    if(isTargetOperation())
    {
        auto targetOperation = value_cast<TargetOperation>(*this);

        for(auto& operand : targetOperation.getInputOperandRange())
        {
            auto targetValue = value_cast<TargetValue>(operand.getValue());

            auto definitions = targetValue.getDefinitions();

            if(definitions.empty())
            {
                continue;
            }

            assert(definitions.size() == 1);

            auto definingOperation = definitions.front().getOperation();

            operations.push_back(definingOperation);
        }
    }
    else
    {
        for(auto& use : getOperands())
        {
            // skip normal values that are not operations
            if(!use.getValue().isOperation())
            {
                continue;
            }

            operations.push_back(Operation(use.getOperation()));
        }
    }

    return operations;
}

OperationList Operation::getSuccessors() const
{
    OperationList operations;

    if(isTargetOperation())
    {
        auto targetOperation = value_cast<TargetOperation>(*this);

        if(targetOperation.hasOutputOperand())
        {
            auto& output = targetOperation.getOutputOperand();

            for(auto& use : output.getValue().getUses())
            {
                auto user = use.getParent().getValue();

                if(user == targetOperation)
                {
                    continue;
                }

                assert(user.isOperation());

                operations.push_back(value_cast<Operation>(user));
            }
        }
    }
    else
    {
        for(auto& use : _implementation->getUses())
        {
            operations.push_back(use.getOperation());
        }
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
    return _implementation->isPHI();
}

bool Operation::isReturn() const
{
    return _implementation->isReturn();
}

bool Operation::isCall() const
{
    return _implementation->isCall();
}

bool Operation::isVoid() const
{
    return getType().isVoid();
}

bool Operation::isTargetOperation() const
{
    return static_cast<bool>(std::dynamic_pointer_cast<TargetOperationImplementation>(
        getValueImplementation()));
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

void Operation::detach()
{
    util::log("Operation") << "Detaching uses of " << toString() << "...\n";
    for(auto use = begin(); use != end(); )
    {
        auto next = use; ++next;
        util::log("Operation") << " detaching use " << use->toString() << "\n";
        use->detach();
        use = next;
    }
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

size_t Operation::getIndexInBasicBlock() const
{
    return std::distance(const_operation_iterator(getBasicBlock().begin()), getIterator());
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

