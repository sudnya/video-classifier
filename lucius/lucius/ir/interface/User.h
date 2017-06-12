/*  \file   User.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the User class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class Value; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing uses of values. */
class User : public Value
{

private:
    typedef std::list<Use*> UseList;

private:
    UseList::iterator _position;
    Value*            _value;
};

} // namespace ir
} // namespace lucius







