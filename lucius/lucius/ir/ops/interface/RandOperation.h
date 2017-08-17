/*  \file   RandOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the RandOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief An operation that produces a random uniform array of data and updates random state. */
class RandOperation : public Operation
{
public:
    RandOperation(Value state, Type tensorType);
    ~RandOperation() final;

public:
    // forward shape operation
    ShapeList getOutputShapes(const ShapeList& inputShapes) const final;

    // backward shape operation
    ShapeList getInputShapes(const ShapeList& outputShapes) const final;

};

} // namespace ir
} // namespace lucius




