/*  \file   CallOperation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the CallOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/CallOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/ExternalFunction.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/ops/implementation/ControlOperationImplementation.h>

// Standard Library Includes
#include <cassert>
#include <sstream>

namespace lucius
{

namespace ir
{

class CallOperationImplementation : public ControlOperationImplementation
{
public:
    virtual ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        if(getType().isVoid())
        {
            return ShapeList();
        }
        else if(getType().isTensor())
        {
            auto& operand = getOperand(0);

            auto operandValue = operand.getValue();

            assert(operandValue.isFunction());

            auto function = value_cast<Function>(operandValue);

            auto value = function.getReturn().getOperand(0).getValue();

            assert(value.getType().isTensor());

            auto tensorType = type_cast<TensorType>(value.getType());

            return {tensorType.getShape()};
        }

        return ShapeList();
    }

    virtual ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return ShapeList();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<CallOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "call";
    }

public:
    virtual Type getType() const
    {
        auto callee = getOperand(0).getValue();

        if(callee.isExternalFunction())
        {
            auto function = value_cast<ExternalFunction>(callee);

            return function.getReturnType();
        }
        else
        {
            auto function = value_cast<Function>(callee);

            return function.getReturnType();
        }
    }

    virtual BasicBlockVector getPossibleTargets() const
    {
        if(getBasicBlock().isExitBlock())
        {
            return {};
        }

        return {getBasicBlock().getNextBasicBlock()};
    }

    virtual bool canFallthrough() const
    {
        return true;
    }

    virtual bool isCall() const
    {
        return true;
    }

public:
    virtual std::string toString() const
    {
        if(getType().isVoid())
        {
            std::stringstream stream;

            stream << name() << "{%" << getId() << "} " << operandString();

            return stream.str();
        }

        return OperationImplementation::toString();
    }

};

CallOperation::CallOperation(Function target)
: ControlOperation(std::make_shared<CallOperationImplementation>())
{
    setOperands({target});
}

CallOperation::CallOperation(ExternalFunction target)
: ControlOperation(std::make_shared<CallOperationImplementation>())
{
    setOperands({target});
}

CallOperation::~CallOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius




