/*  \file   TargetOperationFactory.h
    \author Gregory Diamos
    \date   August 16, 2017
    \brief  The header file for the TargetOperationFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucius { namespace ir { class TargetOperation; } }
namespace lucius { namespace ir { class TargetValue;     } }
namespace lucius { namespace ir { class Type;            } }
namespace lucius { namespace ir { class Context;         } }

namespace lucius
{

namespace ir
{

/*! \brief A factory for creating target operations and operands. */
class TargetOperationFactory
{
public:
    using Context = ir::Context;

public:
    explicit TargetOperationFactory(Context& context);
    virtual ~TargetOperationFactory();

public:
    /*! \brief Create a named operation for this machine. */
    virtual TargetOperation create(const std::string& name) const = 0;

    /*! \brief Create a value of a specified type for this machine. */
    virtual TargetValue createOperand(const Type& t) const = 0;

protected:
    Context& _context;

};

} // namespace ir
} // namespace lucius


