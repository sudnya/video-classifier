/*  \file   TargetOperationImplementation.cpp
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The source file for the TargetOperationImplementation class.
*/

// Lucius Includes
#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/target/interface/TargetValue.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace ir
{

using UseList = std::list<Use>;

TargetOperationImplementation::TargetOperationImplementation()
: _hasOutputOperand(false)
{

}

const Use& TargetOperationImplementation::getOperand(size_t index) const
{
    auto use = getUses().begin();

    std::advance(use, index);

    return *use;
}

Use& TargetOperationImplementation::getOperand(size_t index)
{
    auto use = getUses().begin();

    std::advance(use, index);

    return *use;
}

Use& TargetOperationImplementation::getOutputOperand()
{
    return getUses().back();
}

const Use& TargetOperationImplementation::getOutputOperand() const
{
    return getUses().back();
}

UseList& TargetOperationImplementation::getAllOperands()
{
    return getUses();
}

const UseList& TargetOperationImplementation::getAllOperands() const
{
    return getUses();
}

void TargetOperationImplementation::setOutputOperand(const TargetValue& v)
{
    if(!_hasOutputOperand)
    {
        getUses().push_back(Use(v.getValue()));
        _hasOutputOperand = true;
    }
    else
    {
        getOutputOperand() = Use(v.getValue());
    }
}

void TargetOperationImplementation::_growToSupportIndex(size_t index)
{
    size_t size = getUses().size();

    if(_hasOutputOperand)
    {
        --size;
    }

    for(size_t i = size; i < index; ++i)
    {
        getUses().push_back(Use());
    }
}

void TargetOperationImplementation::setOperand(const TargetValue& v, size_t index)
{
    _growToSupportIndex(index);

    getOperand(index) = Use(v.getValue());
}

void TargetOperationImplementation::appendOperand(const TargetValue& v)
{
    auto end = getUses().end();

    if(_hasOutputOperand)
    {
        assert(!getUses().empty());
        --end;
    }

    getUses().insert(end, Use(v.getValue()));
}

} // namespace ir
} // namespace lucius







