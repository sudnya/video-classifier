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

void OperationImplementation::setOperands(const UseList& uses)
{
    setPredecessorUses(uses);
}

BasicBlock OperationImplementation::getParent() const
{
    return BasicBlock(_parent.lock());
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

            bindToContext(module.getContext());
        }
    }
}

OperationImplementation::operation_iterator OperationImplementation::getIterator()
{
    return _iterator;
}

OperationImplementation::const_operation_iterator OperationImplementation::getIterator() const
{
    return _iterator;
}

std::string OperationImplementation::toString() const
{
    std::stringstream stream;

    stream << "%" << getId() << " = " << name() << " ";

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

std::string OperationImplementation::toSummaryString() const
{
    return toString();
}

} // namespace ir
} // namespace lucius






