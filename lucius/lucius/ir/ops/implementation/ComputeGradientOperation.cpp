/*  \file   ComputeGradientOperation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ComputeGradientOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/ComputeGradientOperation.h>

#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/values/interface/TensorValue.h>

namespace lucius
{

namespace ir
{

ComputeGradientOperation::ComputeGradientOperation(Value value, Value cost)
{
    setOperands({value, cost});
}

ComputeGradientOperation::ComputeGradientOperation(
    std::shared_ptr<ValueImplementation> implementation)
: Operation(implementation)
{

}

ComputeGradientOperation::~ComputeGradientOperation()
{

}

ShapeList ComputeGradientOperation::getOutputShapes(const ShapeList& inputShapes) const
{
    return {inputShapes.front()};
}

ShapeList ComputeGradientOperation::getInputShapes(const ShapeList& outputShapes) const
{
    auto cost = value_cast<TensorValue>(getOperand(1).getValue());

    return {outputShapes.front(), cost.getShape()};
}

} // namespace ir
} // namespace lucius



