/*  \file   UserImplementation.cpp
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The source file for the UserImplementation class.
*/

// Lucius Includes
#include <lucius/ir/implementation/UserImplementation.h>

#include <lucius/ir/interface/Use.h>

namespace lucius
{

namespace ir
{

UserImplementation::UseList& UserImplementation::getPredecessorUses()
{
    return _predecessorUses;
}

const UserImplementation::UseList& UserImplementation::getPredecessorUses() const
{
    return _predecessorUses;
}

void UserImplementation::setPredecessorUses(const UseList& uses)
{
    _predecessorUses = uses;
}

} // namespace ir
} // namespace lucius







