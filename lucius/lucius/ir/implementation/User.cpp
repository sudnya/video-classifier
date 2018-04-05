/*  \file   User.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the User class.
*/

// Lucius Includes
#include <lucius/ir/interface/User.h>

#include <lucius/ir/interface/Value.h>

#include <lucius/ir/implementation/UserImplementation.h>
#include <lucius/ir/implementation/OperationImplementation.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace ir
{

User::User(std::shared_ptr<UserImplementation> implementation)
: _implementation(implementation)
{

}

User::~User()
{

}

Value User::getValue() const
{
    // check if the user is an operation
    auto operation = std::dynamic_pointer_cast<OperationImplementation>(_implementation);

    assert(operation);

    return Value(operation);
}

bool User::isValue() const
{
    return static_cast<bool>(std::dynamic_pointer_cast<OperationImplementation>(_implementation));
}

std::shared_ptr<UserImplementation> User::getImplementation() const
{
    return _implementation;
}

void User::removePredecessorUse(iterator use)
{
    _implementation->removePredecessorUse(use);
}

std::string User::toString() const
{
    if(isValue())
    {
        return getValue().toString();
    }

    assertM(false, "not implemented");
}

} // namespace ir
} // namespace lucius








