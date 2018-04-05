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

namespace lucius
{

namespace ir
{

InsertionPoint getFirstAvailableInsertionPoint(const Value& v);

} // namespace ir
} // namespace lucius




