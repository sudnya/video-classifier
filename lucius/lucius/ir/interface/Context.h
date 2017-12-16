/*  \file   Context.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Context class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class ContextImplementation; } }
namespace lucius { namespace ir { class Type;                  } }
namespace lucius { namespace ir { class Constant;              } }

namespace lucius
{

namespace ir
{

/*! \brief Holds 'global' state for the IR including constants and types. */
class Context
{
public:
    Context();
    ~Context();

public:
    Constant addConstant(Constant c);

public:
    Type addType(Type t);

public:
    size_t allocateId();

public:
    static Context& getDefaultContext();

private:
    std::unique_ptr<ContextImplementation> _implementation;

};

} // namespace ir
} // namespace lucius




