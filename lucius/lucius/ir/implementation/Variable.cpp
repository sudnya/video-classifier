/*  \file   Variable.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Variable class.
*/

// Lucius Includes
#include <lucius/ir/interface/Variable.h>

#include <lucius/ir/implementation/ValueImplementation.h>

namespace lucius
{

namespace ir
{

Variable::Variable()
{
    getValueImplementation()->setIsVariable(true);
}

Variable::Variable(const Value& v)
: Value(v)
{
    getValueImplementation()->setIsVariable(true);
}

} // namespace ir
} // namespace lucius





