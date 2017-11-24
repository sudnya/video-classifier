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

namespace lucius
{

namespace ir
{

class ModuleImplementation
{
public:
    ModuleImplementation(Context& context)
    : _context(&context)
    {

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

} // namespace ir
} // namespace lucius

