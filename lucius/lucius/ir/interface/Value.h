/*  \file   Value.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Value class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Type.h>

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class Module;    } }
namespace lucius { namespace ir { class Use;       } }
namespace lucius { namespace ir { class Operation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a value in a program. */
class Value
{
public:
    Value();
    ~Value();

public:
    void registerWithModule(Module& module);

public:
    Operation getDefinition();
    const Operation getDefinition() const;

private:
    typedef std::list<Use> UseList;

private:
    Type _type;

private:
    UseList _uses;

};

} // namespace ir
} // namespace lucius



