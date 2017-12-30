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
#include <string>
#include <sstream>

namespace lucius
{

namespace ir
{

using UseList = std::list<Use>;

TargetOperationImplementation::TargetOperationImplementation()
: _hasOutputOperand(false)
{

}

Use& TargetOperationImplementation::getOutputOperand()
{
    return getOperands().back();
}

const Use& TargetOperationImplementation::getOutputOperand() const
{
    return getOperands().back();
}

void TargetOperationImplementation::setOutputOperand(const TargetValue& v)
{
    if(!_hasOutputOperand)
    {
        getOperands().push_back(Use(v.getValue()));
        _hasOutputOperand = true;
    }
    else
    {
        getOutputOperand() = Use(v.getValue());
    }
}

void TargetOperationImplementation::setOperand(const TargetValue& v, size_t index)
{
    _growToSupportIndex(index);

    getOperand(index) = Use(v.getValue());
}

void TargetOperationImplementation::appendOperand(const TargetValue& v)
{
    auto end = getOperands().end();

    if(_hasOutputOperand)
    {
        assert(!getOperands().empty());
        --end;
    }

    getOperands().insert(end, Use(v.getValue()));
}

void TargetOperationImplementation::_growToSupportIndex(size_t index)
{
    size_t size = getOperands().size();

    if(_hasOutputOperand)
    {
        --size;
    }

    for(size_t i = size; i < index; ++i)
    {
        getOperands().push_back(Use());
    }
}

} // namespace ir
} // namespace lucius







