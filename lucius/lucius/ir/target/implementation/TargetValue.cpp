/*  \file   TargetValue.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the TargetValue class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/TargetUse.h>

#include <lucius/ir/interface/Type.h>

namespace lucius
{

namespace ir
{

using TargetUseList = TargetValue::TargetUseList;

class TargetValueImplementation
{
public:
    TargetValueImplementation()
    {

    }

    ~TargetValueImplementation()
    {

    }

public:
    Type& getType()
    {
        return _type;
    }

    const Type& getType() const
    {
        return _type;
    }

public:
    TargetUseList& getUses()
    {
        return _uses;
    }

    TargetUseList& getDefinitions()
    {
        return _definitions;
    }

    size_t getId() const
    {
        return _id;
    }

private:
    size_t _id;

private:
    Type _type;

private:
    TargetUseList _uses;
    TargetUseList _definitions;
};

TargetValue::TargetValue()
: _implementation(std::make_shared<TargetValueImplementation>())
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

TargetValue::TargetUseList& TargetValue::getUses()
{
    return _implementation->getUses();
}

TargetValue::TargetUseList& TargetValue::getDefinitions()
{
    return _implementation->getDefinitions();
}

bool TargetValue::operator==(const TargetValue& v) const
{
    return _implementation->getId() == v._implementation->getId();
}

} // namespace ir
} // namespace lucius



