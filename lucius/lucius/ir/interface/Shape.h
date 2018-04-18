/*  \file   Shape.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the Shape class.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <string>

// Forward Declarations
namespace lucius { namespace matrix { class Dimension;           } }
namespace lucius { namespace ir     { class ShapeImplementation; } }

namespace lucius
{
namespace ir
{

/*! \brief A shape represents the dimensions of a tensor in the IR. */
class Shape
{
public:
    Shape();
    Shape(std::initializer_list<size_t>);
    Shape(const matrix::Dimension& d);
    Shape(const Shape& );
    ~Shape();

public:
    Shape& operator=(const Shape& );

public:
    /*! \brief Indicate that the shape is completely unknown. */
    void setUnknown();

    /*! \brief Are dimensions known. */
    bool areAllDimensionsKnown() const;

public:
    using       iterator = size_t*;
    using const_iterator = const size_t*;

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

public:
    /*! \brief Get a reference to the index. */
    size_t& operator[](size_t i);
    size_t  operator[](size_t i) const;

public:
    /*! \brief Get the number of dimensions. */
    size_t size() const;

public:
    /*! \brief Get the total number of elements, if known. */
    size_t elements() const;

public:
    std::string toString() const;

public:
    matrix::Dimension getDimension() const;

private:
    std::unique_ptr<ShapeImplementation> _implementation;

};

} // namespace ir
} // namespace lucius

