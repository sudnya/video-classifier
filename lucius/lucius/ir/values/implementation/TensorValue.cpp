/*  \file   TensorValue.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the TensorValue class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/TensorValue.h>

#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/implementation/ValueImplementation.h>

namespace lucius
{

namespace ir
{

class TensorValueImplementation : public ValueImplementation
{
public:
    using Precision = matrix::Precision;

public:
    TensorValueImplementation(const Shape& s, const Precision& p)
    : _type(TensorType(s, p))
    {

    }

public:
    const Shape& getShape() const
    {
        return _type.getShape();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<TensorValueImplementation>(*this);
    }

    std::string toString() const
    {
        return "Tensor(" + getType().toString() + ")" + "%" + std::to_string(getId());
    }

public:
    Type getType() const
    {
        return _type;
    }

private:
    TensorType _type;
};

TensorValue::TensorValue(const Shape& s, const Precision& p)
: TensorValue(std::make_shared<TensorValueImplementation>(s, p))
{

}

TensorValue::TensorValue(std::shared_ptr<ValueImplementation> implementation)
: Value(implementation)
{

}

const Shape& TensorValue::getShape() const
{
    return std::static_pointer_cast<TensorValueImplementation>(
        getValueImplementation())->getShape();
}

} // namespace ir
} // namespace lucius








