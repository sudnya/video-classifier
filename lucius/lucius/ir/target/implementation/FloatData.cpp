/*  \file   FloatData.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the FloatData class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/FloatData.h>

#include <lucius/ir/target/implementation/TargetValueDataImplementation.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace ir
{

class FloatDataImplementation : public TargetValueDataImplementation
{
public:
    FloatDataImplementation(float value)
    : _value(value)
    {

    }

public:
    float getFloat() const
    {
        return _value;
    }

public:
    virtual void* getData() const
    {
        return const_cast<void*>(reinterpret_cast<const void*>(&_value));
    }

private:
    float _value;

};

FloatData::FloatData(float value)
: FloatData(std::make_shared<FloatDataImplementation>(value))
{

}

FloatData::FloatData()
: FloatData(std::make_shared<FloatDataImplementation>(0))
{

}

FloatData::FloatData(std::shared_ptr<TargetValueDataImplementation> implementation)
: _implementation(std::static_pointer_cast<FloatDataImplementation>(implementation))
{
    assert(static_cast<bool>(std::dynamic_pointer_cast<FloatDataImplementation>(
        implementation)));
}

float FloatData::getFloat() const
{
    return _implementation->getFloat();
}

std::shared_ptr<TargetValueDataImplementation> FloatData::getImplementation() const
{
    return _implementation;
}

} // namespace ir
} // namespace lucius











