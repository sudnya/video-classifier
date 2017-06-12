/*  \file   Value.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Value class.
*/

#pragma once

// Standard Library Includes
#include <list>
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class Type;   } }
namespace lucius { namespace ir { class Module; } }
namespace lucius { namespace ir { class Use;    } }

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
    void registerWithModule(Module* module);

public:
    Operation* getDefinition();
    const Operation* getDefinition() const;

private:
    typedef std::list<Use*> UseList;
    typedef std::list<std::unique_ptr<Value>> ValueList;

private:
    Type* _type;

private:
    UseList _uses;

private:
    ValueList::iterator _position;
    Module*             _module;

};

} // namespace ir
} // namespace lucius



