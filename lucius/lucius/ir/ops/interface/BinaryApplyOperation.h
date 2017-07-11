/*  \file   BinaryApplyOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the BinaryApplyOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operation. */
class BinaryApplyOperation : public Operation
{
public:
    BinaryApplyOperation(Value left, Value right, Value operation);
    ~BinaryApplyOperation() final;

public:
    // forward shape operation
    ShapeList getOutputShapes(const ShapeList& inputShapes) const final;

    // backward shape operation
    ShapeList getInputShapes(const ShapeList& outputShapes) const final;

};

} // namespace ir
} // namespace lucius





