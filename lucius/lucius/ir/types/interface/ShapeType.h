/*  \file   ShapeType.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ShapeType class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Type.h>

// Forward Declarations
namespace lucius { namespace ir { class Shape;              } }
namespace lucius { namespace ir { class TypeImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a shape type. */
class ShapeType : public Type
{
public:
    ShapeType(const Shape& d);
    explicit ShapeType(std::shared_ptr<TypeImplementation> );
    ~ShapeType();

public:
    const Shape& getShape() const;

};

} // namespace ir
} // namespace lucius




