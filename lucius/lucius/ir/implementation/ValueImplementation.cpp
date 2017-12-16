/*  \file   ValueImplementation.cpp
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the ValueImplementation class.
*/

// Lucius Includes
#include <lucius/ir/implementation/ValueImplementation.h>

#include <lucius/ir/implementation/OperationImplementation.h>
#include <lucius/ir/implementation/ConstantImplementation.h>
#include <lucius/ir/implementation/FunctionImplementation.h>

#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Context.h>

namespace lucius
{

namespace ir
{

ValueImplementation::ValueImplementation()
: _isVariable(false), _context(nullptr)
{

}

ValueImplementation::~ValueImplementation()
{

}

ValueImplementation::UseList& ValueImplementation::getUses()
{
    return _uses;
}

const ValueImplementation::UseList& ValueImplementation::getUses() const
{
    return _uses;
}

size_t ValueImplementation::getId() const
{
    return _id;
}

bool ValueImplementation::isOperation() const
{
    return dynamic_cast<const OperationImplementation*>(this);
}

bool ValueImplementation::isConstant() const
{
    return dynamic_cast<const ConstantImplementation*>(this);
}

bool ValueImplementation::isFunction() const
{
    return dynamic_cast<const FunctionImplementation*>(this);
}

bool ValueImplementation::isVariable() const
{
    return _isVariable;
}

void ValueImplementation::setIsVariable(bool b)
{
    _isVariable = b;
}

void ValueImplementation::bindToContext(Context& context)
{
    _context = &context;

    _id = _context->allocateId();
}

} // namespace ir
} // namespace lucius





