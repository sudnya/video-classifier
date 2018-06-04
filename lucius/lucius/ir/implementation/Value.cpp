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
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Variable.h>
#include <lucius/ir/interface/ExternalFunction.h>

#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/TargetOperation.h>

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

Value::Value(Function f)
: _implementation(f.getValueImplementation())
{

}

Value::Value(ExternalFunction f)
: _implementation(f.getValueImplementation())
{

}

Value::Value(TargetValue t)
: _implementation(t.getValueImplementation())
{

}

Value::Value(TargetOperation t)
: _implementation(t.getValueImplementation())
{

}

Value::Value(Variable v)
: Value(v.getValue())
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

bool Value::isValid() const
{
    return static_cast<bool>(_implementation);
}

bool Value::isOperation() const
{
    return _implementation->isOperation();
}

bool Value::isTargetOperation() const
{
    return _implementation->isTargetOperation();
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

bool Value::isExternalFunction() const
{
    return _implementation->isExternalFunction();
}

bool Value::isFunction() const
{
    return _implementation->isFunction();
}

bool Value::isTensor() const
{
    return getType().isTensor();
}

bool Value::isInteger() const
{
    return getType().isInteger();
}

bool Value::isFloat() const
{
    return getType().isFloat();
}

bool Value::isPointer() const
{
    return getType().isPointer();
}

bool Value::isBasicBlock() const
{
    return getType().isBasicBlock();
}

bool Value::isRandomState() const
{
    return getType().isRandomState();
}

bool Value::isStructure() const
{
    return getType().isStructure();
}

bool Value::isShape() const
{
    return getType().isShape();
}

bool Value::isTargetValue() const
{
    return _implementation->isTargetValue();
}

Value::UseList& Value::getUses()
{
    return _implementation->getUses();
}

const Value::UseList& Value::getUses() const
{
    return _implementation->getUses();
}

void Value::addUse(const Use& u)
{
    getUses().push_back(u);
    getUses().back().setValuePosition(--getUses().end());
}

void Value::removeUse(use_iterator u)
{
    getUses().erase(u);
}

void Value::bindToContext(Context* context)
{
    _implementation->bindToContext(context);
}

void Value::bindToContextIfDifferent(Context* context)
{
    _implementation->bindToContextIfDifferent(context);
}

Context& Value::getContext()
{
    return *_implementation->getContext();
}

bool Value::operator<(const Value& right) const
{
    return _implementation->getId() < right._implementation->getId();
}

bool Value::operator==(const Value& right) const
{
    return _implementation->getId() == right._implementation->getId();
}

std::shared_ptr<ValueImplementation> Value::getValueImplementation() const
{
    return _implementation;
}

std::string Value::toString() const
{
    return _implementation->toString();
}

std::string Value::toSummaryString() const
{
    return _implementation->toSummaryString();
}

} // namespace ir
} // namespace lucius




