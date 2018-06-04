/*  \file   TargetValue.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the TargetValue class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/TargetValue.h>

#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/User.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/implementation/OperationImplementation.h>

#include <lucius/ir/values/interface/ConstantInteger.h>

#include <lucius/machine/generic/interface/IntegerData.h>

#include <lucius/ir/target/implementation/TargetValueImplementation.h>

#include <lucius/matrix/interface/Operator.h>
#include <lucius/matrix/interface/Matrix.h>

// Standard Library Includes
#include <string>
#include <cassert>

namespace lucius
{

namespace ir
{

static bool isTargetImplementation(std::shared_ptr<ValueImplementation> i)
{
    return static_cast<bool>(std::dynamic_pointer_cast<TargetValueImplementation>(i));
}

TargetValue::TargetValue()
{

}

TargetValue::TargetValue(std::shared_ptr<ValueImplementation> implementation)
: _implementation(implementation)
{

}

TargetValue::TargetValue(Value v)
: TargetValue(v.getValueImplementation())
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

const TargetValue::UseList& TargetValue::getUses() const
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

    static UseList dummyDefinitions;

    return dummyDefinitions;
}

const TargetValue::UseList& TargetValue::getDefinitions() const
{
    if(isTargetImplementation(getValueImplementation()))
    {
        return std::static_pointer_cast<TargetValueImplementation>(
            _implementation)->getDefinitions();
    }

    static UseList dummyDefinitions;

    return dummyDefinitions;
}

TargetValue::UseList TargetValue::getUsesAndDefinitions() const
{
    auto result = getUses();
    auto definitions = getDefinitions();

    result.insert(result.end(), definitions.begin(), definitions.end());

    return result;
}

void TargetValue::addDefinition(const Use& u)
{
    assert(isTargetImplementation(getValueImplementation()));

    auto targetValueImplementation = std::static_pointer_cast<TargetValueImplementation>(
        _implementation);

    targetValueImplementation->getDefinitions().push_back(u);
}

void TargetValue::allocateData()
{
    assert(isTargetImplementation(getValueImplementation()));

    auto targetValueImplementation = std::static_pointer_cast<TargetValueImplementation>(
        _implementation);

    targetValueImplementation->allocateData();
}

void TargetValue::freeData()
{
    assert(isTargetImplementation(getValueImplementation()));

    auto targetValueImplementation = std::static_pointer_cast<TargetValueImplementation>(
        _implementation);

    targetValueImplementation->freeData();
}

bool TargetValue::isAllocated() const
{
    assert(isTargetImplementation(getValueImplementation()));

    auto data = std::static_pointer_cast<TargetValueImplementation>(_implementation)->getData();

    return data.isValid();
}

Value TargetValue::getValue() const
{
    return Value(_implementation);
}

bool TargetValue::isValid() const
{
    return static_cast<bool>(_implementation);
}

bool TargetValue::isVariable() const
{
    return getValue().isVariable();
}

bool TargetValue::isConstant() const
{
    return _implementation->isConstant();
}

bool TargetValue::isOperation() const
{
    return getValue().isOperation();
}

bool TargetValue::isTensor() const
{
    return getValue().isTensor();
}

bool TargetValue::isInteger() const
{
    return getValue().isInteger();
}

bool TargetValue::isFloat() const
{
    return getValue().isFloat();
}

bool TargetValue::isPointer() const
{
    return getValue().isPointer();
}

bool TargetValue::isRandomState() const
{
    return getValue().isRandomState();
}

bool TargetValue::isStructure() const
{
    return getValue().isStructure();
}

bool TargetValue::isShape() const
{
    return getValue().isShape();
}

bool TargetValue::operator==(const TargetValue& v) const
{
    return _implementation->getId() == v._implementation->getId();
}

bool TargetValue::operator<(const TargetValue& v) const
{
    return _implementation->getId() < v._implementation->getId();
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

    if(isConstant())
    {
        if(isInteger())
        {
            auto constantInteger = value_cast<ConstantInteger>(getValue());

            return machine::generic::IntegerData(constantInteger.getValue());
        }
    }

    return TargetValueData();
}

void TargetValue::setData(const TargetValueData& data)
{
    assert(isTargetImplementation(getValueImplementation()));

    return std::static_pointer_cast<TargetValueImplementation>(_implementation)->setData(data);
}

Context& TargetValue::getContext()
{
    return getValue().getContext();
}

std::shared_ptr<UserImplementation> TargetValue::getUserImplementation() const
{
    if(isOperation())
    {
        auto operation = std::static_pointer_cast<OperationImplementation>(
            getValueImplementation());

        return operation;
    }

    assert(false);
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



