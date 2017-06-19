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

class UseImplementation
{
private:
    using UseList = std::list<Use>;

private:
    std::weak_ptr<User>  _parent;
    std::weak_ptr<Value> _value;

private:
    UseList::iterator _parentPosition;

private:
    UseList::iterator _valuePosition;

};

/*! \brief A class for representing and tracking the use of a value. */
class Use
{

private:
    std::shared_ptr<UseImplementation> _implementation;
};

} // namespace ir
} // namespace lucius






