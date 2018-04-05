/*  \file   TargetValueImplementation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the TargetValueImplementation class.
*/

// Lucius Includes
#include <lucius/ir/target/implementation/TargetValueImplementation.h>

namespace lucius
{

namespace ir
{

TargetValueImplementation::UseList& TargetValueImplementation::getDefinitions()
{
    return _definitions;
}

const TargetValueImplementation::UseList& TargetValueImplementation::getDefinitions() const
{
    return _definitions;
}

TargetValueData TargetValueImplementation::getData() const
{
    return _data;
}

void TargetValueImplementation::setData(TargetValueData data)
{
    _data = data;
}

} // namespace ir
} // namespace lucius






