/*  \file   BinaryApplyOperation.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the BinaryApplyOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/BinaryApplyOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>

namespace lucius
{

namespace ir
{

BinaryApplyOperation::BinaryApplyOperation(Value left, Value right, Value operation)
{
    setOperands({left, right, operation});
}

BinaryApplyOperation::~BinaryApplyOperation()
{
    // intentionally blank
}

ShapeList BinaryApplyOperation::getOutputShapes(const ShapeList& inputShapes) const
{
    return {inputShapes.front()};
}

ShapeList BinaryApplyOperation::getInputShapes(const ShapeList& outputShapes) const
{
    return {outputShapes.front(), outputShapes.front()};
}

} // namespace ir
} // namespace lucius






