/*  \file   OperationImplementation.h
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The header file for the OperationImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/ValueImplementation.h>

namespace lucius
{

namespace ir
{

/*! \brief The implementation of a class that represents an operation in the program. */
class OperationImplementation : public ValueImplementation
{
public:
    // forward shape operation
    virtual ShapeList getOutputShapes(const ShapeList& inputShapes) const;

    // backward shape operation
    virtual ShapeList getInputShapes(const ShapeList& outputShapes) const;

public:
    const Use& getOperand(size_t index) const;
          Use& getOperand(size_t index);

    const Shape& getOperandShape(size_t index) const;
          Shape& getOperandShape(size_t index);

};

} // namespace ir
} // namespace lucius





