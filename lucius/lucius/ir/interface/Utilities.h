/*  \file   Utilities.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the IR Utilities set of functions.
*/

#pragma once

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class Value;          } }
namespace lucius { namespace ir { class InsertionPoint; } }
namespace lucius { namespace ir { class Operation;      } }
namespace lucius { namespace ir { class Program;        } }

namespace lucius
{

namespace ir
{

/*! \brief Get the first point in the program where the value is available. */
InsertionPoint getFirstAvailableInsertionPoint(const Value& v);

/*! \brief Get the last valid location to insert operations in the program. */
InsertionPoint getProgramExitPoint(const Program& v);

/*! \brief Modify the block so that it is legal to insert a terminator at the specified point. */
InsertionPoint prepareBlockToAddTerminator(const InsertionPoint& point);

using OperationList = std::list<Operation>;
using const_operation_iterator = OperationList::const_iterator;

/*! \brief Move operations from one block to another. */
void moveOperations(const InsertionPoint& point,
    const_operation_iterator begin, const_operation_iterator end);

} // namespace ir
} // namespace lucius




