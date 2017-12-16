/*  \file   ComputeGradientOperationImplementation.cpp
    \author Gregory Diamos
    \date   October 12, 2017
    \brief  The source file for the ComputeGradientOperationImplementation class.
*/

// Lucius Includes
#include <lucius/ir/ops/implementation/ComputeGradientOperationImplementation.h>

#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/values/interface/TensorValue.h>

#include <lucius/ir/implementation/OperationImplementation.h>

// Standard Library Includes
#include <string>

namespace lucius
{

namespace ir
{

ShapeList ComputeGradientOperationImplementation::getOutputShapes(
    const ShapeList& inputShapes) const
{
    return {inputShapes.front()};
}

ShapeList ComputeGradientOperationImplementation::getInputShapes(
    const ShapeList& outputShapes) const
{
    auto cost = value_cast<TensorValue>(getOperand(1).getValue());

    return {outputShapes.front(), cost.getShape()};
}

std::shared_ptr<ValueImplementation> ComputeGradientOperationImplementation::clone() const
{
    return std::make_shared<ComputeGradientOperationImplementation>(*this);
}

std::string ComputeGradientOperationImplementation::name() const
{
    return "gradient";
}

Type ComputeGradientOperationImplementation::getType() const
{
    return getOperand(0).getValue().getType();
}

} // namespace ir
} // namespace lucius


