/*  \file   PointerData.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the PointerData class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/PointerData.h>

#include <lucius/ir/target/implementation/TargetValueDataImplementation.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{
namespace machine
{
namespace generic
{

class PointerDataImplementation : public ir::TargetValueDataImplementation
{
public:
    PointerDataImplementation(void* value)
    : _value(value)
    {

    }

public:
    void* getPointer() const
    {
        return _value;
    }

public:
    virtual void* getData() const
    {
        return const_cast<void*>(reinterpret_cast<const void*>(&_value));
    }

private:
    void* _value;

};

PointerData::PointerData(void* value)
: PointerData(std::make_shared<PointerDataImplementation>(value))
{

}

PointerData::PointerData()
: PointerData(std::make_shared<PointerDataImplementation>(nullptr))
{

}

PointerData::PointerData(std::shared_ptr<ir::TargetValueDataImplementation> implementation)
: _implementation(std::static_pointer_cast<PointerDataImplementation>(implementation))
{
    assert(static_cast<bool>(std::dynamic_pointer_cast<PointerDataImplementation>(
        implementation)));
}

void* PointerData::getPointer() const
{
    return _implementation->getPointer();
}

std::shared_ptr<ir::TargetValueDataImplementation> PointerData::getImplementation() const
{
    return _implementation;
}

} // namespace generic
} // namespace machine
} // namespace lucius











