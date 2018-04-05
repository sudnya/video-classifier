/*  \file   ConstantFloat.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ConstantFloat class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/ConstantFloat.h>

#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/ConstantImplementation.h>

// Standard Library Includes
#include <string>

namespace lucius
{

namespace ir
{

class ConstantFloatValueImplementation : public ConstantImplementation
{
public:
    ConstantFloatValueImplementation(float value)
    : _value(value)
    {

    }

    float getValue() const
    {
        return _value;
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ConstantFloatValueImplementation>(*this);
    }

    std::string toString() const
    {
        return std::to_string(_value);
    }

    Type getType() const
    {
        return Type(Type::FloatId);
    }

private:
    float _value;
};

ConstantFloat::ConstantFloat(std::shared_ptr<ValueImplementation> implementation)
: Constant(implementation)
{

}

ConstantFloat::ConstantFloat(float value)
: Constant(std::make_shared<ConstantFloatValueImplementation>(value))
{

}

float ConstantFloat::getValue() const
{
    return std::static_pointer_cast<ConstantFloatValueImplementation>(
        getValueImplementation())->getValue();
}

} // namespace ir
} // namespace lucius









