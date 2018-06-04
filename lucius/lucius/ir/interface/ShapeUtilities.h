/*  \file   ShapeUtilities.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the IR ShapeUtilities set of functions.
*/

#pragma once

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class Shape; } }

namespace lucius
{

namespace ir
{

/*! \brief Remove the named elements from the shape. */
Shape removeElements(const Shape& shapeToRemoveFrom, const Shape& elementsToRemove);

} // namespace ir
} // namespace lucius





