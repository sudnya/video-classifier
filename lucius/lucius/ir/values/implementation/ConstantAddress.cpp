/*  \file   ConstantAddress.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ConstantAddress class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/ConstantAddress.h>

#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/ConstantImplementation.h>

// Standard Library Includes
#include <string>
#include <sstream>

namespace lucius
{

namespace ir
{

class ConstantAddressValueImplementation : public ConstantImplementation
{
public:
    ConstantAddressValueImplementation(void* value)
    : _value(value)
    {

    }

    void* getAddress() const
    {
        return _value;
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ConstantAddressValueImplementation>(*this);
    }

    std::string toString() const
    {
        std::stringstream stream;

        stream << _value;

        return stream.str();
    }

    Type getType() const
    {
        return Type(Type::PointerId);
    }

private:
    void* _value;
};

ConstantAddress::ConstantAddress(std::shared_ptr<ValueImplementation> implementation)
: Constant(implementation)
{

}

ConstantAddress::ConstantAddress(void* value)
: Constant(std::make_shared<ConstantAddressValueImplementation>(value))
{

}

void* ConstantAddress::getAddress() const
{
    return std::static_pointer_cast<ConstantAddressValueImplementation>(
        getValueImplementation())->getAddress();
}

} // namespace ir
} // namespace lucius









