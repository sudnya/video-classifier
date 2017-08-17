/*  \file   Module.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Module class.
*/

// Lucius Includes
#include <lucius/ir/interface/Module.h>

#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Variable.h>

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

private:
    Context* _context;

private:
    using FunctionList = std::list<Function>;
    using VariableList = std::list<Variable>;

private:
    FunctionList _functions;
    VariableList _variables;
};

Module::Module(Context& context)
: _implementation(std::make_shared<ModuleImplementation>(context))
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

} // namespace ir
} // namespace lucius

