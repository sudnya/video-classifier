/*  \file   Value.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Value class.
*/

// Lucius Includes
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Constant.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Gradient.h>
#include <lucius/ir/interface/Function.h>

#include <lucius/ir/implementation/ValueImplementation.h>

namespace lucius
{

namespace ir
{

Value::Value()
{

}

Value::Value(std::shared_ptr<ValueImplementation> implementation)
: _implementation(implementation)
{

}

Value::Value(Operation o)
: _implementation(o.getValueImplementation())
{

}

Value::Value(Constant c)
: _implementation(c.getValueImplementation())
{

}

Value::Value(BasicBlock b)
: _implementation(b.getValueImplementation())
{

}

Value::Value(Gradient g)
: _implementation(g.getValueImplementation())
{

}

Value::Value(Function f)
: _implementation(f.getValueImplementation())
{

}

Value::~Value()
{

}

Value::Value(const Value& v)
: _implementation(v.getValueImplementation())
{

}

Value& Value::operator=(const Value& v)
{
    _implementation = v.getValueImplementation();

    return *this;
}

Type Value::getType() const
{
    return _implementation->getType();
}

bool Value::isOperation() const
{
    return _implementation->isOperation();
}

bool Value::isConstant() const
{
    return _implementation->isConstant();
}

bool Value::isVoid() const
{
    return getType().isVoid();
}

bool Value::isVariable() const
{
    return _implementation->isVariable();
}

bool Value::isFunction() const
{
    return _implementation->isFunction();
}

Value::UseList& Value::getUses()
{
    return _implementation->getUses();
}

const Value::UseList& Value::getUses() const
{
    return _implementation->getUses();
}

bool Value::operator<(const Value& right) const
{
    return _implementation->getId() < right._implementation->getId();
}

std::shared_ptr<ValueImplementation> Value::getValueImplementation() const
{
    return _implementation;
}

std::string Value::toString() const
{
    return _implementation->toString();
}

} // namespace ir
} // namespace lucius




