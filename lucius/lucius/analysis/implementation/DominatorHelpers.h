/*  \file   DominatorHelpers.h
    \author Gregory Diamos
    \date   December 8, 2017
    \brief  The header file for the set of dominator algorithm helper functions.
*/

#pragma once

// Standard Library Includes
#include <map>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlock; } }
namespace lucius { namespace ir { class Function;   } }

namespace lucius
{

namespace analysis
{

using BasicBlock = ir::BasicBlock;
using Function = ir::Function;
using BasicBlockMap = std::map<BasicBlock, BasicBlock>;
using BlockIndexMap = std::map<BasicBlock, size_t>;

using BlockVector = std::vector<BasicBlock>;
using BlockVectorMap = std::map<BasicBlock, BlockVector>;

void buildDominatorTree(BasicBlockMap& tree, BlockIndexMap& positions,
    const Function& function, bool isDominatorAnalysis);

BasicBlock intersect(BasicBlock left, BasicBlock right,
    const BlockIndexMap& positions, const BasicBlockMap& dominatorTree);

void buildDominanceFrontiers(BlockVectorMap& dominanceFrontiers,
    const BasicBlockMap& dominatorTree, bool isDominatorAnalysis);


} // namespace analysis
} // namespace lucius




