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

#include <lucius/ir/implementation/ValueImplementation.h>

namespace lucius
{

namespace ir
{

Value::Value()
: _implementation(std::make_shared<ValueImplementation>())
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

const Type& Value::getType() const
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

} // namespace ir
} // namespace lucius




