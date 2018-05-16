/*  \file   Module.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Module class.
*/

// Lucius Includes
#include <lucius/ir/interface/Module.h>

#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Variable.h>
#include <lucius/ir/interface/Context.h>

// Standard Library Includes
#include <sstream>

namespace lucius
{

namespace ir
{

class ModuleImplementation
{
public:
    ModuleImplementation(Context& context)
    : _name("UnknownModuleName"), _context(&context)
    {

    }

    ~ModuleImplementation()
    {
        _removeSelfReferences();
    }

public:
    Function addFunction(Function f)
    {
        _functions.push_back(f);

        return _functions.back();
    }

public:
    Variable addVariable(Variable variable)
    {
        _variables.push_back(variable);

        return _variables.back();
    }

public:
    Context& getContext()
    {
        return *_context;
    }

public:
    using FunctionList = std::list<Function>;
    using VariableList = std::list<Variable>;

public:
    FunctionList::iterator begin()
    {
        return _functions.begin();
    }

    FunctionList::const_iterator begin() const
    {
        return _functions.begin();
    }

    FunctionList::iterator end()
    {
        return _functions.end();
    }

    FunctionList::const_iterator end() const
    {
        return _functions.end();
    }

public:
    void clear()
    {
        _functions.clear();
        _variables.clear();
    }

public:
    bool containsFunction(const std::string& name) const
    {
        // TODO: make this more efficient

        for(auto& function : *this)
        {
            if(function.name() == name)
            {
                return true;
            }
        }

        return false;
    }

    Function getFunction(const std::string& name) const
    {
        // TODO: make this more efficient

        for(auto& function : *this)
        {
            if(function.name() == name)
            {
                return function;
            }
        }

        throw std::runtime_error("Requested function " + name + " does not exist." );
    }

public:
    VariableList& getVariables()
    {
        return _variables;
    }

    const VariableList& getVariables() const
    {
        return _variables;
    }

public:
    const std::string& getName() const
    {
        return _name;
    }

    void setName(const std::string& name)
    {
        _name = name;
    }

private:
    void _removeSelfReferences()
    {
        // TODO: make sure that nothing in the module holds a self reference
    }

private:
    std::string _name;

private:
    Context* _context;

private:
    FunctionList _functions;
    VariableList _variables;
};

Module::Module(Context& context)
: _implementation(std::make_shared<ModuleImplementation>(context))
{

}

Module::Module(std::shared_ptr<ModuleImplementation> implementation)
: _implementation(implementation)
{

}

Module::Module()
: Module(Context::getDefaultContext())
{

}

Module::~Module()
{
    // intentionally blank
}

Function Module::addFunction(Function f)
{
    f.setParent(*this);

    return _implementation->addFunction(f);
}

Variable Module::addVariable(Variable v)
{
    return _implementation->addVariable(v);
}

Context& Module::getContext()
{
    return _implementation->getContext();
}

Module::iterator Module::begin()
{
    return _implementation->begin();
}

Module::const_iterator Module::begin() const
{
    return _implementation->begin();
}

Module::iterator Module::end()
{
    return _implementation->end();
}

Module::const_iterator Module::end() const
{
    return _implementation->end();
}

void Module::clear()
{
    _implementation->clear();
}

bool Module::containsFunction(const std::string& name) const
{
    return _implementation->containsFunction(name);
}

Function Module::getFunction(const std::string& name) const
{
    return _implementation->getFunction(name);
}

Module::VariableList& Module::getVariables()
{
    return _implementation->getVariables();
}

const Module::VariableList& Module::getVariables() const
{
    return _implementation->getVariables();
}

std::string Module::toString() const
{
    std::stringstream stream;

    stream << "// Variables\n";

    for(auto& variable : getVariables())
    {
        stream << variable.toString() << "\n";
    }

    stream << "// Functions\n";

    for(auto& function : *this)
    {
        stream << function.toString() << "\n";
    }

    return stream.str();
}

void Module::setName(const std::string& name)
{
    _implementation->setName(name);
}

const std::string& Module::name() const
{
    return _implementation->getName();
}

bool Module::operator==(const Module& m) const
{
    return name() == m.name();
}

bool Module::operator!=(const Module& m) const
{
    return !(*this == m);
}

std::shared_ptr<ModuleImplementation> Module::getImplementation()
{
    return _implementation;
}

} // namespace ir
} // namespace lucius

