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

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/implementation/OperationImplementation.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace ir
{

class CallOperationImplementation : public OperationImplementation
{
public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
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

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return ShapeList();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<CallOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "call";
    }

public:
    Type getType() const
    {
        return getOperand(0).getValue().getType();
    }

};

CallOperation::CallOperation(Function target)
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




