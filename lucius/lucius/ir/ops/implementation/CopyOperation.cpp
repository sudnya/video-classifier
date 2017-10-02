/*  \file   CopyOperation.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the CopyOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/CopyOperation.h>

#include <lucius/ir/interface/ShapeList.h>

namespace lucius
{

namespace ir
{

CopyOperation::CopyOperation(Value input)
{
    setOperands({input});
}

CopyOperation::~CopyOperation()
{
    // intentionally blank
}

ShapeList CopyOperation::getOutputShapes(const ShapeList& inputShapes) const
{
    return inputShapes;
}

ShapeList CopyOperation::getInputShapes(const ShapeList& outputShapes) const
{
    return outputShapes;
}

} // namespace ir
} // namespace lucius


