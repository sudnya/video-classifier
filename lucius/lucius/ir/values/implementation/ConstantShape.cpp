/*  \file   ConstantShape.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ConstantShape class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/ConstantShape.h>

#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/ConstantImplementation.h>

namespace lucius
{

namespace ir
{

class ConstantShapeImplementation : public ConstantImplementation
{
public:
    ConstantShapeImplementation(const matrix::Dimension& s)
    : _shape(s)
    {

    }

    virtual ~ConstantShapeImplementation()
    {

    }

public:
    Shape& getContents()
    {
        return _shape;
    }

    const Shape& getContents() const
    {
        return _shape;
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ConstantShapeImplementation>(*this);
    }

public:
    std::string toString() const
    {
        return "shape(" + _shape.toString() + ")";
    }

public:
    Type getType() const
    {
        return Type(Type::ArrayId);
    }

private:
    Shape _shape;

};

ConstantShape::ConstantShape(const matrix::Dimension& value)
: Constant(std::make_shared<ConstantShapeImplementation>(value))
{

}

ConstantShape::ConstantShape(std::shared_ptr<ValueImplementation> implementation)
: Constant(implementation)
{

}

Shape& ConstantShape::getContents()
{
    return std::static_pointer_cast<ConstantShapeImplementation>(
        getValueImplementation())->getContents();
}

const Shape& ConstantShape::getContents() const
{
    return std::static_pointer_cast<ConstantShapeImplementation>(
        getValueImplementation())->getContents();
}

} // namespace ir
} // namespace lucius








