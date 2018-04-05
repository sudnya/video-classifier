/*  \file   ConstantPointer.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ConstantPointer class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/ConstantPointer.h>

#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/ConstantImplementation.h>

// Standard Library Includes
#include <string>
#include <sstream>

namespace lucius
{

namespace ir
{

class ConstantPointerValueImplementation : public ConstantImplementation
{
public:
    ConstantPointerValueImplementation(void* value)
    : _value(value)
    {

    }

    void* getValue() const
    {
        return _value;
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ConstantPointerValueImplementation>(*this);
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

ConstantPointer::ConstantPointer(std::shared_ptr<ValueImplementation> implementation)
: Constant(implementation)
{

}

ConstantPointer::ConstantPointer(void* value)
: Constant(std::make_shared<ConstantPointerValueImplementation>(value))
{

}

void* ConstantPointer::getValue() const
{
    return std::static_pointer_cast<ConstantPointerValueImplementation>(
        getValueImplementation())->getValue();
}

} // namespace ir
} // namespace lucius









