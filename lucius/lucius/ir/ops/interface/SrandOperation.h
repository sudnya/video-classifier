/*  \file   SrandOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the SrandOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief An operation that converts a seed into a random generator state. */
class SrandOperation : public Operation
{
public:
    SrandOperation(Value seed);
    ~SrandOperation() final;

public:
    // forward shape operation
    ShapeList getOutputShapes(const ShapeList& inputShapes) const final;

    // backward shape operation
    ShapeList getInputShapes(const ShapeList& outputShapes) const final;

};

} // namespace ir
} // namespace lucius



