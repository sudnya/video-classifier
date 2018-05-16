/*  \file   IntegerData.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the IntegerData class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/IntegerData.h>

#include <lucius/ir/target/implementation/TargetValueDataImplementation.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{
namespace machine
{
namespace generic
{

class IntegerDataImplementation : public ir::TargetValueDataImplementation
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

    void setInteger(size_t value)
    {
        _value = value;
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

IntegerData::IntegerData(std::shared_ptr<ir::TargetValueDataImplementation> implementation)
: _implementation(std::static_pointer_cast<IntegerDataImplementation>(implementation))
{
    assert(static_cast<bool>(std::dynamic_pointer_cast<IntegerDataImplementation>(
        implementation)));
}

size_t IntegerData::getInteger() const
{
    return _implementation->getInteger();
}

void IntegerData::setInteger(size_t value)
{
    return _implementation->setInteger(value);
}

std::shared_ptr<ir::TargetValueDataImplementation> IntegerData::getImplementation() const
{
    return _implementation;
}

} // namespace generic
} // namespace machine
} // namespace lucius










