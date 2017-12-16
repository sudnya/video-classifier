/*  \file   Traversals.h
    \author Gregory Diamos
    \date   December 8, 2017
    \brief  The header file for the set of IR traversal functions.
*/

#pragma once

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlock; } }

namespace lucius
{

namespace analysis
{

using BasicBlockList = std::list<ir::BasicBlock>;

BasicBlockList reversePostOrderTraversal(const BasicBlockList& blocks, bool reverseEdges);

BasicBlockList postOrderTraversal(const BasicBlockList& blocks, bool reverseEdges);

} // namespace analysis
} // namespace lucius



