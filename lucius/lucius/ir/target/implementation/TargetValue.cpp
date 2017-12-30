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

#include <lucius/ir/target/interface/TargetValueData.h>

#include <lucius/ir/implementation/ValueImplementation.h>

// Standard Library Includes
#include <string>

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

public:
    TargetValueData getData() const
    {
        return _data;
    }

private:
    UseList _definitions;

private:
    TargetValueData _data;
};

static bool isTargetImplementation(std::shared_ptr<ValueImplementation> i)
{
    return static_cast<bool>(std::dynamic_pointer_cast<TargetValueImplementation>(i));
}

TargetValue::TargetValue()
{

}

TargetValue::TargetValue(Value v)
: _implementation(v.getValueImplementation())
{

}

TargetValue::~TargetValue()
{
    // intentionally blank
}

Type TargetValue::getType() const
{
    return _implementation->getType();
}

TargetValue::UseList& TargetValue::getUses()
{
    return _implementation->getUses();
}

TargetValue::UseList& TargetValue::getDefinitions()
{
    if(isTargetImplementation(getValueImplementation()))
    {
        return std::static_pointer_cast<TargetValueImplementation>(
            _implementation)->getDefinitions();
    }

    static TargetValue::UseList dummyDefinitions;

    return dummyDefinitions;
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

TargetValueData TargetValue::getData() const
{
    if(isTargetImplementation(getValueImplementation()))
    {
        return std::static_pointer_cast<TargetValueImplementation>(_implementation)->getData();
    }

    return TargetValueData();
}

std::shared_ptr<ValueImplementation> TargetValue::getValueImplementation() const
{
    return _implementation;
}

std::string TargetValue::toString() const
{
    return _implementation->toString();
}

} // namespace ir
} // namespace lucius



