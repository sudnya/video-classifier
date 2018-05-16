/*  \file   OperationImplementation.cpp
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The source file for the OperationImplementation class.
*/

// Lucius Includes
#include <lucius/ir/implementation/OperationImplementation.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Module.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/User.h>
#include <lucius/ir/interface/Value.h>

// Standard Library Includes
#include <cassert>
#include <string>
#include <sstream>

namespace lucius
{

namespace ir
{

ShapeList OperationImplementation::getOutputShapes(const ShapeList& inputShapes) const
{
    // Null operations do nothing
    return {};
}

ShapeList OperationImplementation::getInputShapes(const ShapeList& outputShapes) const
{
    // Null operations do nothing
    return {};
}

const Use& OperationImplementation::getOperand(size_t index) const
{
    auto use = getPredecessorUses().begin();

    std::advance(use, index);

    return *use;
}

Use& OperationImplementation::getOperand(size_t index)
{
    auto use = getPredecessorUses().begin();

    std::advance(use, index);

    return *use;
}

const Shape& OperationImplementation::getOperandShape(size_t index) const
{
    auto& operand = getOperand(index);

    auto value = operand.getValue();

    assert(value.getType().isTensor());

    auto tensorType = type_cast<TensorType>(value.getType());

    return tensorType.getShape();
}

const OperationImplementation::UseList& OperationImplementation::getOperands() const
{
    return getPredecessorUses();
}

OperationImplementation::UseList& OperationImplementation::getOperands()
{
    return getPredecessorUses();
}

OperationImplementation::iterator OperationImplementation::begin()
{
    return getOperands().begin();
}

OperationImplementation::const_iterator OperationImplementation::begin() const
{
    return getOperands().begin();
}

OperationImplementation::iterator OperationImplementation::end()
{
    return getOperands().end();
}

OperationImplementation::const_iterator OperationImplementation::end() const
{
    return getOperands().end();
}

size_t OperationImplementation::size() const
{
    return getOperands().size();
}

bool OperationImplementation::empty() const
{
    return getOperands().empty();
}

void OperationImplementation::setOperands(const UseList& uses)
{
    setPredecessorUses(uses);

    for(auto position = begin(); position != end(); ++position)
    {
        auto& use = *position;

        use.setParent(User(getImplementation()), position);
        use.getValue().addUse(use);

        use.getValue().bindToContextIfDifferent(getContext());
    }
}

void OperationImplementation::setOperands(const ValueList& values)
{
    UseList list;

    for(auto& value : values)
    {
        list.push_back(Use(value));
    }

    setOperands(list);
}

void OperationImplementation::appendOperand(const Use& use)
{
    insertOperand(end(), use);
}

void OperationImplementation::appendOperand(const Value& value)
{
    appendOperand(Use(value));
}

void OperationImplementation::replaceOperand(const Use& original, const Use& newOperand)
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

void OperationImplementation::insertOperand(iterator position, const Use& operand)
{
    auto newPosition = getOperands().insert(position, operand);

    newPosition->setParent(User(getImplementation()), newPosition);
    newPosition->getValue().addUse(*newPosition);

    newPosition->getValue().bindToContextIfDifferent(getContext());
}

BasicBlock OperationImplementation::getParent() const
{
    return BasicBlock(_parent.lock());
}

BasicBlock OperationImplementation::getBasicBlock() const
{
    return getParent();
}

void OperationImplementation::setParent(const BasicBlock& parent)
{
    _parent = parent.getImplementation();

    if(parent.hasParent())
    {
        auto function = parent.getParent();

        if(function.hasParent())
        {
            auto module = function.getParent();

            bindToContextIfDifferent(&module.getContext());

            for(auto& operand : getOperands())
            {
                operand.getValue().bindToContextIfDifferent(&module.getContext());
            }
        }
    }
}

void OperationImplementation::setIterator(operation_iterator it)
{
    _iterator = it;
}

OperationImplementation::operation_iterator OperationImplementation::getIterator()
{
    return _iterator;
}

OperationImplementation::const_operation_iterator OperationImplementation::getIterator() const
{
    return _iterator;
}

void OperationImplementation::setImplementation(
    std::weak_ptr<OperationImplementation> implementation)
{
    _this = implementation;
}

std::shared_ptr<OperationImplementation> OperationImplementation::getImplementation() const
{
    return _this.lock();
}

std::string OperationImplementation::toString() const
{
    std::stringstream stream;

    stream << "%" << getId() << " = " << name() << " " << operandString();

    return stream.str();
}

std::string OperationImplementation::toSummaryString() const
{
    std::stringstream stream;

    stream << getType().toString() << " %" << getId();

    return stream.str();
}

std::string OperationImplementation::operandString() const
{
    std::stringstream stream;

    auto operandIterator = getOperands().begin();

    if(operandIterator != getOperands().end())
    {
        auto& operand = *operandIterator;

        stream << operand.getValue().toSummaryString();

        ++operandIterator;
    }

    for( ; operandIterator != getOperands().end(); ++operandIterator)
    {
        auto& operand = *operandIterator;

        stream << ", " << operand.getValue().toSummaryString();
    }

    return stream.str();
}

} // namespace ir
} // namespace lucius






