/*  \file   Module.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Module class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class ModuleImplementation; } }
namespace lucius { namespace ir { class Context;              } }
namespace lucius { namespace ir { class Constant;             } }
namespace lucius { namespace ir { class Type;                 } }
namespace lucius { namespace ir { class Variable;             } }

namespace lucius
{

namespace ir
{


/*! \brief A class for representing a module. */
class Module
{
public:
    Module(Context& context);
    ~Module();

public:
    Function addFunction(const std::string& name);

public:
    Constant addConstant(Constant c);

public:
    Type addType(Type t);

public:
    Variable addVariable(Variable v);

private:
    std::shared_ptr<ModuleImplementation> _implementation;
};

} // namespace ir
} // namespace lucius




