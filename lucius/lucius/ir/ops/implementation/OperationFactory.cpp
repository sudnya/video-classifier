/*  \file   OperationFactory.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the OperationFactory class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/OperationFactory.h>

#include <lucius/ir/ops/interface/BroadcastOperation.h>
#include <lucius/ir/ops/interface/SetOperation.h>

namespace lucius
{

namespace ir
{

constexpr bool DoesntNeedSavedOutput = false;

Operation OperationFactory::createBackPropagationOperation(const Operation& op)
{
    if(op.name() == "reduce")
    {
        return BroadcastOperation();
    }
    else if(op.name() == "get")
    {
        return SetOperation();
    }

    throw std::runtime_error("TODO: add back propagation operation for '" + op.toString() + "'");
}

OperationFactory::BackPropagationOperandDescriptor
    OperationFactory::getBackPropagationOperandDescriptor(const Operation& op)
{
    if(op.name() == "reduce")
    {
        return BackPropagationOperandDescriptor(DoesntNeedSavedOutput, {1, 2});
    }
    else if(op.name() == "get")
    {
        return BackPropagationOperandDescriptor(DoesntNeedSavedOutput, {1});
    }

    throw std::runtime_error("TODO: add back propagation operand descriptor for '" +
        op.toString() + "'");
}

using BackPropagationOperandDescriptor = OperationFactory::BackPropagationOperandDescriptor;

BackPropagationOperandDescriptor::BackPropagationOperandDescriptor(bool needsSavedOutput,
    const IndexVector& savedIndices)
: _needsSavedOutput(needsSavedOutput), _savedIndices(savedIndices)
{

}

bool BackPropagationOperandDescriptor::needsSavedOutput() const
{
    return _needsSavedOutput;
}

BackPropagationOperandDescriptor::iterator BackPropagationOperandDescriptor::begin()
{
    return _savedIndices.begin();
}

BackPropagationOperandDescriptor::const_iterator BackPropagationOperandDescriptor::begin() const
{
    return _savedIndices.begin();
}

BackPropagationOperandDescriptor::iterator BackPropagationOperandDescriptor::end()
{
    return _savedIndices.end();
}

BackPropagationOperandDescriptor::const_iterator BackPropagationOperandDescriptor::end() const
{
    return _savedIndices.end();
}

} // namespace ir
} // namespace lucius




