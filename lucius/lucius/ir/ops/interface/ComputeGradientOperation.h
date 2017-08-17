/*  \file   ComputeGradientOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ComputeGradientOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a gradient value with respect to a cost value. */
class ComputeGradientOperation : public Operation
{
public:
    ComputeGradientOperation(Value value, Value cost);
    explicit ComputeGradientOperation(std::shared_ptr<ValueImplementation>);
    ~ComputeGradientOperation() final;

public:
    // forward shape operation
    ShapeList getOutputShapes(const ShapeList& inputShapes) const final;

    // backward shape operation
    ShapeList getInputShapes(const ShapeList& outputShapes) const final;

};

} // namespace ir
} // namespace lucius


