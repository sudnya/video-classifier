/*  \file   Use.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Use class.
*/

// Lucius Includes
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/interface/User.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

class UseImplementation
{
public:
    UseImplementation(const Value& value)
    : _value(value.getValueImplementation())
    {

    }

public:
    Value getValue() const
    {
        return Value(_value);
    }

    User getUser() const
    {
        return User(_parent.lock());
    }

public:
    Operation getOperation() const
    {
        return Operation(_value);
    }

    BasicBlock getParent() const
    {
       return BasicBlock(_parent.lock());
    }

private:
    using UseList = std::list<Use>;

private:
    std::weak_ptr<UserImplementation>  _parent;
    std::shared_ptr<ValueImplementation> _value;

private:
    UseList::iterator _parentPosition;

private:
    UseList::iterator _valuePosition;

};

Use::Use()
{

}

Use::Use(const Value& value)
: _implementation(std::make_shared<UseImplementation>(value))
{

}

Use::~Use()
{
    // intetionally blank
}

BasicBlock Use::getParent() const
{
    return _implementation->getParent();
}

Operation Use::getOperation() const
{
    return _implementation->getOperation();
}

Value Use::getValue() const
{
    return _implementation->getValue();
}

} // namespace ir
} // namespace lucius








