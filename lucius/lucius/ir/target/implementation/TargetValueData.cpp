/*  \file   TargetValueData.cpp
    \author Gregory Diamos
    \date   December 19, 2017
    \brief  The source file for the TargetValueData class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/TargetValueData.h>

#include <lucius/ir/target/implementation/TargetValueDataImplementation.h>

namespace lucius
{

namespace ir
{

TargetValueData::TargetValueData(std::shared_ptr<TargetValueDataImplementation> implementation)
: _implementation(implementation)
{

}

TargetValueData::TargetValueData()
{
    // intentionally blank
}

std::shared_ptr<TargetValueDataImplementation> TargetValueData::getImplementation() const
{
    return _implementation;
}

void* TargetValueData::data() const
{
    return _implementation->getData();
}

bool TargetValueData::isValid() const
{
    return static_cast<bool>(_implementation);
}

} // namespace ir
} // namespace lucius








