/*  \file   Module.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Module class.
*/

#pragma once

// Standard Library Includes
#include <list>
#include <string>

// Forward Declarations
namespace lucius { namespace ir { class ModuleImplementation; } }
namespace lucius { namespace ir { class Context;              } }
namespace lucius { namespace ir { class Constant;             } }
namespace lucius { namespace ir { class Type;                 } }
namespace lucius { namespace ir { class Variable;             } }
namespace lucius { namespace ir { class Function;             } }

namespace lucius
{

namespace ir
{


/*! \brief A class for representing a module. */
class Module
{
public:
    Module(Context& context);
    Module(std::shared_ptr<ModuleImplementation> module);
    Module();
    ~Module();

public:
    Function addFunction(Function f);

public:
    Variable addVariable(Variable v);

public:
    Context& getContext();

public:
    bool containsFunction(const std::string& name) const;

public:
    Function getFunction(const std::string& name) const;

public:
    using FunctionList = std::list<Function>;
    using iterator = FunctionList::iterator;
    using const_iterator = FunctionList::const_iterator;

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

public:
    using VariableList = std::list<Variable>;

public:
    VariableList& getVariables();
    const VariableList& getVariables() const;

public:
    void clear();

public:
    void setName(const std::string& name);
    const std::string& name() const;

public:
    std::string toString() const;

public:
    bool operator==(const Module&) const;
    bool operator!=(const Module&) const;

public:
    std::shared_ptr<ModuleImplementation> getImplementation();

private:
    std::shared_ptr<ModuleImplementation> _implementation;
};

} // namespace ir
} // namespace lucius




