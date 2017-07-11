/*  \file   ApplyOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ApplyOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operation. */
class ApplyOperation : public Operation
{
public:
    ApplyOperation(Value input, Value operation);
    ~ApplyOperation() final;

public:
    // forward shape operation
    ShapeList getOutputShapes(const ShapeList& inputShapes) const final;

    // backward shape operation
    ShapeList getInputShapes(const ShapeList& outputShapes) const final;

};

} // namespace ir
} // namespace lucius




