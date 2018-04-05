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
    using UseList = std::list<Use>;
    using iterator = UseList::iterator;

public:
    Value getValue() const
    {
        return Value(_value);
    }

    void setValuePosition(iterator position)
    {
        _valuePosition = position;
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

    BasicBlock getBasicBlock() const
    {
        return getOperation().getParent();
    }

    void setParent(std::weak_ptr<UserImplementation> parent, iterator position)
    {
        _parent = parent;
        _parentPosition = position;
    }

    User getParent() const
    {
        return User(_parent.lock());
    }

public:
    void detach()
    {
        getParent().removePredecessorUse(_parentPosition);
        getValue().removeUse(_valuePosition);
    }

private:
    std::weak_ptr<UserImplementation>    _parent;
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

BasicBlock Use::getBasicBlock() const
{
    return _implementation->getBasicBlock();
}

User Use::getParent() const
{
    return _implementation->getParent();
}

void Use::setParent(User parent, iterator position)
{
    _implementation->setParent(parent.getImplementation(), position);
}

void Use::detach()
{
    _implementation->detach();
}

Operation Use::getOperation() const
{
    return _implementation->getOperation();
}

Value Use::getValue() const
{
    return _implementation->getValue();
}

void Use::setValuePosition(iterator position)
{
    _implementation->setValuePosition(position);
}

std::string Use::toString() const
{
    return getValue().toString();
}

} // namespace ir
} // namespace lucius








