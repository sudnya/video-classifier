/*  \file   ShapeUtilities.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the IR ShapeUtilities set of functions.
*/

// Lucius Includes
#include <lucius/ir/interface/ShapeUtilities.h>

#include <lucius/ir/interface/Shape.h>

#include <lucius/matrix/interface/DimensionTransformations.h>

namespace lucius
{

namespace ir
{

/*! \brief Remove the named elements from the shape. */
Shape removeElements(const Shape& shapeToRemoveFrom, const Shape& elementsToRemove)
{
    return Shape(removeDimensions(shapeToRemoveFrom.getDimension(),
                                  elementsToRemove.getDimension()));
}

} // namespace ir
} // namespace lucius






