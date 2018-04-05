/*  \file   ExternalFunctionImplementation.cpp
    \author Gregory Diamos
    \date   March 17, 2018
    \brief  The source file for the ExternalFunctionImplementation class.
*/

// Lucius Includes
#include <lucius/ir/implementation/ExternalFunctionImplementation.h>

// Standard Library Includes
#include <string>
#include <sstream>

namespace lucius
{

namespace ir
{

ExternalFunctionImplementation::ExternalFunctionImplementation(
    const std::string& name, const TypeVector& types)
: _name(name), _argumentTypes(types), _passArgumentsAsTargetValues(false)
{

}

const std::string& ExternalFunctionImplementation::getName() const
{
    return _name;
}

std::shared_ptr<ValueImplementation> ExternalFunctionImplementation::clone() const
{
    return std::make_shared<ExternalFunctionImplementation>(*this);
}

std::string ExternalFunctionImplementation::toString() const
{
    return toSummaryString();
}

void ExternalFunctionImplementation::setPassArgumentsAsTargetValues(bool condition)
{
    _passArgumentsAsTargetValues = condition;
}

bool ExternalFunctionImplementation::getPassArgumentsAsTargetValues() const
{
    return _passArgumentsAsTargetValues;
}

std::string ExternalFunctionImplementation::toSummaryString() const
{
    std::stringstream result;

    // TODO: Support return values
    result << "external void " << getName() << "{" << getId() << "} (";

    auto type = _argumentTypes.begin();

    if(type != _argumentTypes.end())
    {
        result << type->toString();
        ++type;
    }

    for( ; type != _argumentTypes.end(); ++type)
    {
        result << ", " << type->toString();
    }

    result << ")";

    return result.str();
}

Type ExternalFunctionImplementation::getType() const
{
    return Type(Type::FunctionId);
}

} // namespace ir
} // namespace lucius









