/*  \file   ConstantInteger.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ConstantInteger class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/ConstantInteger.h>

#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/ValueImplementation.h>

namespace lucius
{

namespace ir
{

class ConstantIntegerValueImplementation : public ValueImplementation
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








