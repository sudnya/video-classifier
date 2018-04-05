/*  \file   IntegerData.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the IntegerData class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/IntegerData.h>

#include <lucius/ir/target/implementation/TargetValueDataImplementation.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace ir
{

class IntegerDataImplementation : public TargetValueDataImplementation
{
public:
    IntegerDataImplementation(size_t value)
    : _value(value)
    {

    }

public:
    size_t getInteger() const
    {
        return _value;
    }

public:
    virtual void* getData() const
    {
        return const_cast<void*>(reinterpret_cast<const void*>(&_value));
    }

private:
    size_t _value;

};

IntegerData::IntegerData(size_t value)
: IntegerData(std::make_shared<IntegerDataImplementation>(value))
{

}

IntegerData::IntegerData()
: IntegerData(std::make_shared<IntegerDataImplementation>(0))
{

}

IntegerData::IntegerData(std::shared_ptr<TargetValueDataImplementation> implementation)
: _implementation(std::static_pointer_cast<IntegerDataImplementation>(implementation))
{
    assert(static_cast<bool>(std::dynamic_pointer_cast<IntegerDataImplementation>(
        implementation)));
}

size_t IntegerData::getInteger() const
{
    return _implementation->getInteger();
}

std::shared_ptr<TargetValueDataImplementation> IntegerData::getImplementation() const
{
    return _implementation;
}

} // namespace ir
} // namespace lucius










