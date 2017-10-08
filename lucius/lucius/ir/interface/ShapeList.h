/*  \file   ShapeList.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the ShapeList class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class Shape; } }

namespace lucius
{
namespace ir
{

/*! \brief A list of shapes. */
class ShapeList
{
public:
    ShapeList();
    ShapeList(std::initializer_list<Shape>);
    ~ShapeList();

public:
          Shape& front();
    const Shape& front() const;

};

} // namespace ir
} // namespace lucius


