/*  \file   ConstantInteger.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ConstantInteger class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/ConstantInteger.h>

#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/ConstantImplementation.h>

namespace lucius
{

namespace ir
{

class ConstantIntegerValueImplementation : public ConstantImplementation
{
public:
    ConstantIntegerValueImplementation(size_t value)
    : _value(value)
    {

    }

    size_t getValue() const
    {
        return _value;
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ConstantIntegerValueImplementation>(*this);
    }

private:
    size_t _value;
};

ConstantInteger::ConstantInteger(size_t value)
: Constant(std::make_shared<ConstantIntegerValueImplementation>(value))
{

}

size_t ConstantInteger::getValue() const
{
    return std::static_pointer_cast<ConstantIntegerValueImplementation>(
        getValueImplementation())->getValue();
}

} // namespace ir
} // namespace lucius








