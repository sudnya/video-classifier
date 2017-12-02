/*  \file   TargetValue.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the TargetValue class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/TargetValue.h>

#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/implementation/ValueImplementation.h>

namespace lucius
{

namespace ir
{

using UseList = TargetValue::UseList;

class TargetValueImplementation : public ValueImplementation
{
public:
    TargetValueImplementation()
    {

    }

    ~TargetValueImplementation()
    {

    }

public:
    UseList& getDefinitions()
    {
        return _definitions;
    }

private:
    UseList _definitions;
};

TargetValue::TargetValue()
{

}

TargetValue::TargetValue(Value v)
: _implementation(std::static_pointer_cast<TargetValueImplementation>(v.getValueImplementation()))
{

}

TargetValue::~TargetValue()
{
    // intentionally blank
}

Type& TargetValue::getType()
{
    return _implementation->getType();
}

const Type& TargetValue::getType() const
{
    return _implementation->getType();
}

TargetValue::UseList& TargetValue::getUses()
{
    return _implementation->getUses();
}

TargetValue::UseList& TargetValue::getDefinitions()
{
    return _implementation->getDefinitions();
}

Value TargetValue::getValue() const
{
    return Value(_implementation);
}

bool TargetValue::operator==(const TargetValue& v) const
{
    return _implementation->getId() == v._implementation->getId();
}

bool TargetValue::operator==(const Value& v) const
{
    return _implementation->getId() == v.getValueImplementation()->getId();
}

} // namespace ir
} // namespace lucius



