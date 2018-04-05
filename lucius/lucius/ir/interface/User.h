/*  \file   User.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the User class.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class UserImplementation;  } }
namespace lucius { namespace ir { class Value;               } }
namespace lucius { namespace ir { class Use;                 } }

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

public:
    /*! \brief Convert to a value if possible. */
    Value getValue() const;

    /*! \brief Check if the user is a value itself. */
    bool isValue() const;

public:
    using UseList = std::list<Use>;
    using iterator = UseList::iterator;

public:
    void removePredecessorUse(iterator position);

public:
    std::string toString() const;

public:
    std::shared_ptr<UserImplementation> getImplementation() const;

private:
    std::shared_ptr<UserImplementation> _implementation;

};

} // namespace ir
} // namespace lucius







