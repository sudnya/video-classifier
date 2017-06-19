/*  \file   Module.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Module class.
*/

// Lucius Includes
#include <lucius/ir/interface/Module.h>

namespace lucius
{

namespace ir
{

class ModuleImplementation
{
private:
    using FunctionList = std::list<Function>;
    using VariableList = std::list<Variable>;
    using ConstantList = std::list<Constant>;

private:
    FunctionList _functions;
    VariableList _variables;
    ConstantList _constants;
};

Module::Module(Conext& context)
: _context(context)
{

}

Module::~Module()
{
    // intentionally blank
}

Function* Module::addFunction(std::unique_ptr<Function>&& f)
{
    auto name = f->name();

    _functions.emplace(name, std::move(f));
}

void Module::addConstant(std::unique_ptr<Constant>&& c)
{
    auto name = c->name();

    _constants.emplace(name, std::move(c));
}

void Module::addType(std::unique_ptr<Type>&& t)
{
    _context.addType(std::move(t));
}

void Module::addValue(std::unique_ptr<Value>&& v)
{
    auto newValue = _values.insert(_values.end(), std::move(v));

    (*vewValue)->registerWithModule(newValue, this);
}

void Module::addVariable(std::unique_ptr<Variable>&& v)
{
    auto newValue = _variables.insert(_variables.end(), std::move(v));

    (*vewValue)->registerWithModule(newValue, this);
}

} // namespace ir
} // namespace lucius

