/*  \file   Use.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Use class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class Value; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing and tracking the use of a value. */
class Use
{

private:
    typedef std::list<Use*> UseList;

private:
    UseList::iterator _position;
    Value*            _value;
};

} // namespace ir
} // namespace lucius






