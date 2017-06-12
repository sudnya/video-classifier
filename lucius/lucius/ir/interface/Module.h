/*  \file   Module.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Module class.
*/

#pragma once

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a module. */
class Module
{
public:
    Module(Conext& context);
    ~Module();

public:
    Function* addFunction(const std::string& name);

public:
    Constant* addConstant(std::unique_ptr<Constant>&& );

public:
    Type* addType(std::unique_ptr<Type>&& t);

public:
    Value* addValue(std::unique_ptr<Value>&& v);
    Variable* addVariable(std::unique_ptr<Variable>&& v);

private:
    Context& _context;

private:
    ConstantMap _constants;
    FunctionMap _functions;

private:
    ValueList    _values;
    VariableList _variables;

};

} // namespace ir
} // namespace lucius




