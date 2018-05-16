/*  \file   Variable.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Variable class.
*/

// Lucius Includes
#include <lucius/ir/interface/Variable.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/implementation/ValueImplementation.h>

// Standard Library Includes
#include <string>

namespace lucius
{

namespace ir
{

class VariableImplementation
{
public:
    VariableImplementation()
    {

    }

    VariableImplementation(const Value& v)
    : _value(v)
    {

    }

public:
    bool isMappedToMachineVariable() const
    {
        return static_cast<bool>(_mappedMachineVariable);
    }

    Variable getMappedMachineVariable() const
    {
        return Variable(_mappedMachineVariable);
    }

    void setMappedMachineVariable(Variable machineVariable)
    {
        _mappedMachineVariable = machineVariable.getVariableImplementation();
    }

public:
    Value getValue() const
    {
        return _value;
    }

private:
    Value _value;

private:
    std::shared_ptr<VariableImplementation> _mappedMachineVariable;

};

Variable::Variable()
: _implementation(std::make_shared<VariableImplementation>())
{

}

Variable::Variable(const Value& v)
: _implementation(std::make_shared<VariableImplementation>(v))
{
    getValue().getValueImplementation()->setIsVariable(true);
}

Variable::Variable(std::shared_ptr<VariableImplementation> implementation)
: _implementation(implementation)
{

}

bool Variable::isMappedToMachineVariable() const
{
    return getVariableImplementation()->isMappedToMachineVariable();
}

Variable Variable::getMappedMachineVariable() const
{
    return getVariableImplementation()->getMappedMachineVariable();
}

void Variable::setMappingToMachineVariable(Variable machineVariable)
{
    getVariableImplementation()->setMappedMachineVariable(machineVariable);
}

Value Variable::getValue() const
{
    return getVariableImplementation()->getValue();
}

std::string Variable::toString() const
{
    return "variable " + getValue().toString();
}

bool Variable::operator<(const Variable& right) const
{
    return getValue() < right.getValue();
}

bool Variable::operator==(const Variable& right) const
{
    return getValue() == right.getValue();
}

std::shared_ptr<VariableImplementation> Variable::getVariableImplementation() const
{
    return _implementation;
}

} // namespace ir
} // namespace lucius





