/*  \file   ExternalFunction.h
    \author Gregory Diamos
    \date   March 17, 2018
    \brief  The header file for the ExternalFunction class.
*/

// Lucius Includes
#include <lucius/ir/interface/ExternalFunction.h>

#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/ExternalFunctionImplementation.h>

// Standard Library Includes
#include <string>
#include <sstream>

namespace lucius
{

namespace ir
{

ExternalFunction::ExternalFunction(const std::string& name, const TypeVector& argumentTypes)
: _implementation(std::make_shared<ExternalFunctionImplementation>(name, argumentTypes))
{

}

ExternalFunction::ExternalFunction(std::shared_ptr<ValueImplementation> implementation)
: _implementation(std::static_pointer_cast<ExternalFunctionImplementation>(implementation))
{

}

ExternalFunction::~ExternalFunction()
{
    // intentionally blank
}

const std::string& ExternalFunction::name() const
{
    return _implementation->getName();
}

std::string ExternalFunction::toString() const
{
    return _implementation->toString();
}

Type ExternalFunction::getReturnType() const
{
    // TODO: support more return types than void
    return Type(Type::VoidId);
}

std::shared_ptr<ExternalFunctionImplementation> ExternalFunction::getImplementation() const
{
    return _implementation;
}

std::shared_ptr<ValueImplementation> ExternalFunction::getValueImplementation() const
{
    return _implementation;
}

void ExternalFunction::setPassArgumentsAsTargetValues(bool condition)
{
    getImplementation()->setPassArgumentsAsTargetValues(condition);
}

bool ExternalFunction::getPassArgumentsAsTargetValues() const
{
    return getImplementation()->getPassArgumentsAsTargetValues();
}

} // namespace ir
} // namespace lucius







