/*  \file   ApplyOperation.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the ApplyOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/ApplyOperation.h>

namespace lucius
{

namespace ir
{

ApplyOperation::ApplyOperation(Value input, Value operation)
{
    setOperands({input, operation});
}

ApplyOperation::~ApplyOperation()
{

}

ApplyOperation::ShapeList ApplyOperation::getOutputShapes(const ShapeList& inputShapes) const
{
    return inputShapes;
}

ApplyOperation::ShapeList ApplyOperation::getInputShapes(const ShapeList& outputShapes) const
{
    return outputShapes;
}

} // namespace ir
} // namespace lucius





