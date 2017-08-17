/*  \file   ControlOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ControlOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a branch. */
class ControlOperation : public Operation
{
public:
    ControlOperation();
    virtual ~ControlOperation();

public:
    // forward shape operation
    ShapeList getOutputShapes(const ShapeList& inputShapes) const;

    // backward shape operation
    ShapeList getInputShapes(const ShapeList& outputShapes) const;


};

} // namespace ir
} // namespace lucius




