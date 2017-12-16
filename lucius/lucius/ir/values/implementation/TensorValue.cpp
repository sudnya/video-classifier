/*  \file   TensorValue.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the TensorValue class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/TensorValue.h>

#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/ValueImplementation.h>

namespace lucius
{

namespace ir
{

class TensorValueImplementation : public ValueImplementation
{
public:
    Shape& getShape()
    {
        return _shape;
    }

    const Shape& getShape() const
    {
        return _shape;
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<TensorValueImplementation>(*this);
    }

    std::string toString() const
    {
        return "Tensor(" + _shape.toString() + ")" + "%" + std::to_string(getId());
    }

public:
    Type getType() const
    {
        return Type(Type::TensorId);
    }

private:
    Shape _shape;
};

TensorValue::TensorValue(std::shared_ptr<ValueImplementation> implementation)
: Value(implementation)
{

}

Shape& TensorValue::getShape()
{
    return std::static_pointer_cast<TensorValueImplementation>(
        getValueImplementation())->getShape();
}

const Shape& TensorValue::getShape() const
{
    return std::static_pointer_cast<TensorValueImplementation>(
        getValueImplementation())->getShape();
}

} // namespace ir
} // namespace lucius








