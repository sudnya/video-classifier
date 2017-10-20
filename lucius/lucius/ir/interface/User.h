/*  \file   User.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the User class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class UserImplementation;  } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a user of other values. */
class User
{
public:
    User(std::shared_ptr<UserImplementation>);
    ~User();

private:
    std::shared_ptr<UserImplementation> _implementation;

};

} // namespace ir
} // namespace lucius







