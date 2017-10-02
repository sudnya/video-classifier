/*  \file   BroadcastOperation.h
    \author Gregory Diamos
    \date   October 2, 2017
    \brief  The source file for the BroadcastOperation class.
*/

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

BroadcastOperation::BroadcastOperation(Value left, Value right, Value dimensions, Value operation)
{
    setOperands({left, right, dimensions, operation});
}

BroadcastOperation::~BroadcastOperation()
{

}

ShapeList BroadcastOperation::getOutputShapes(const ShapeList& inputShapes) const
{
    return {inputShapes.front()};
}

ShapeList BroadcastOperation::getInputShapes(const ShapeList& outputShapes) const
{
    auto dimensionsValue = getOperand(2).getValue();

    auto rightShape = outputShapes.front();

    if(dimensionsValue.isConstant())
    {
        auto dimensions = value_cast<ConstantTensor>(dimensionsValue);

        for(auto& index : dimensions)
        {
            rightShape[index] = 1;
        }
    }
    else
    {
        rightShape.setUnknown();
    }

    return {outputShapes.front(), rightShape};
}

} // namespace ir
} // namespace lucius








