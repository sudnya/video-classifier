/*  \file   Use.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Use class.
*/

// Lucius Includes
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/interface/User.h>
#include <lucius/ir/interface/Value.h>

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

} // namespace ir
} // namespace lucius








