/*  \file   Shape .h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the Shape class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Dimension; } }

namespace lucius
{
namespace ir
{

/*! \brief A shape represents the dimensions of a tensor in the IR. */
class Shape
{
public:
    Shape();
    Shape(const matrix::Dimension& d);

public:
    /*! \brief Indicate that the shape is completely unknown. */
    void setUnknown();

public:
    /*! \brief Get a reference to the index. */
          size_t& operator[](size_t i);
    const size_t& operator[](size_t i) const;

};

} // namespace ir
} // namespace lucius

