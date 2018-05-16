/*  \file   Variable.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Variable class.
*/

#pragma once

// Lucius Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class VariableImplementation; } }
namespace lucius { namespace ir { class Value;                  } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a value corresponding to model parameters in a program.

    IR variables do not store data for the variable in order to maintain different storage
    mechanisms for difference target machines.

    They can be mapped to machine variables which implement data storate.
*/
class Variable
{
public:
    Variable();
    Variable(const Value& v);
    Variable(std::shared_ptr<VariableImplementation>);

public:
    bool isMappedToMachineVariable() const;
    Variable getMappedMachineVariable() const;

    void setMappingToMachineVariable(Variable machineVariable);

public:
    bool operator<(const Variable& right) const;
    bool operator==(const Variable& right) const;

public:
    Value getValue() const;

public:
    std::string toString() const;

public:
    std::shared_ptr<VariableImplementation> getVariableImplementation() const;

private:
    std::shared_ptr<VariableImplementation> _implementation;

};

} // namespace ir
} // namespace lucius




