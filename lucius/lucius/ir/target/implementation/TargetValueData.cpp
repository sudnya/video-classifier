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

void* TargetValueData::data() const
{
    return _implementation->getData();
}

} // namespace ir
} // namespace lucius








