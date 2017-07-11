/*  \file   ReduceOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ReduceOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a reduction operation. */
class ReduceOperation : public Operation
{
public:
    ReduceOperation(Value input, Value dimensions, Value operation);
    ~ReduceOperation() final;

public:
    // forward shape operation
    ShapeList getOutputShapes(const ShapeList& inputShapes) const final;

    // backward shape operation
    ShapeList getInputShapes(const ShapeList& outputShapes) const final;

};

} // namespace ir
} // namespace lucius






