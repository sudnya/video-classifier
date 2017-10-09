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

#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class ComputeGradientOperationImplementation : public OperationImplementation
{
public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        return {inputShapes.front()};
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        auto cost = value_cast<TensorValue>(getOperand(1).getValue());

        return {outputShapes.front(), cost.getShape()};
    }

};

ComputeGradientOperation::ComputeGradientOperation(Value value, Value cost)
: Operation(std::make_shared<ComputeGradientOperationImplementation>())
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

} // namespace ir
} // namespace lucius



