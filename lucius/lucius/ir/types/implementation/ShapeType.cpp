/*  \file   ShapeType.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ShapeType class.
*/

// Lucius Includes
#include <lucius/ir/types/interface/ShapeType.h>

#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/implementation/TypeImplementation.h>


namespace lucius
{

namespace ir
{

class ShapeTypeImplementation : public TypeImplementation
{
public:
    ShapeTypeImplementation(const Shape& shape)
    : TypeImplementation(Type::ShapeId), _shape(shape)
    {

    }

    const Shape& getShape() const
    {
        return _shape;
    }

public:
    virtual std::string toString() const
    {
        return "shape(" + _shape.toString() + ")";
    }

private:
    Shape _shape;

};

ShapeType::ShapeType(std::shared_ptr<TypeImplementation> implementation)
: Type(implementation)
{

}

ShapeType::ShapeType(const Shape& d)
: Type(std::make_shared<ShapeTypeImplementation>(d))
{

}

ShapeType::~ShapeType()
{

}

const Shape& ShapeType::getShape() const
{
    auto implementation = std::static_pointer_cast<ShapeTypeImplementation>(
        getTypeImplementation());

    return implementation->getShape();
}

} // namespace ir
} // namespace lucius





