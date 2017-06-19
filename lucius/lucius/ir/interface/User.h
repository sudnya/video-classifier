/*  \file   User.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the User class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class Use; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a user of other values. */
class User : public Value
{
private:
    using UseList = std::list<Use>;

private:
    UseList _usedValues;
};

} // namespace ir
} // namespace lucius







