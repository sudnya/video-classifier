/*  \file   DominatorHelpers.cpp
    \author Gregory Diamos
    \date   December 8, 2017
    \brief  The source file for the set of dominator algorithm helper functions.
*/

// Lucius Includes
#include <lucius/analysis/implementation/DominatorHelpers.h>

#include <lucius/analysis/interface/Traversals.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Function.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace analysis
{

void buildDominatorTree(BasicBlockMap& tree, BlockIndexMap& positions,
    const Function& function, bool isDominatorAnalysis)
{
    tree.clear();
    positions.clear();

    if(function.empty())
    {
        return;
    }

    auto traversal = reversePostOrderTraversal(
        BasicBlockList(function.begin(), function.end()), !isDominatorAnalysis);

    for(auto& block : traversal)
    {
        positions[block] = positions.size();
    }

    bool changed = true;

    auto exitBlock = traversal.front();
    traversal.pop_front();

    tree[exitBlock] = exitBlock;

    while(changed)
    {
        changed = false;

        for(auto& block : traversal)
        {
            // perform post-dominator set intersections

            // find the first processed predecessor
            auto nextBlocks = isDominatorAnalysis ?
                block.getPredecessors() : block.getSuccessors();

            assert(!nextBlocks.empty());

            BasicBlock newImmediateDominator = *nextBlocks.begin();

            assert(tree.count(newImmediateDominator) != 0);

            // intersect
            for(auto& nextBlock : nextBlocks)
            {
                newImmediateDominator = intersect(nextBlock, newImmediateDominator,
                    positions, tree);
            }

            auto dominator = tree.find(block);

            if(dominator == tree.end())
            {
                tree.insert(std::make_pair(block, newImmediateDominator));
                changed = true;
            }
            else if(dominator->second != newImmediateDominator)
            {
                dominator->second = newImmediateDominator;
                changed = true;
            }
        }
    }

}

BasicBlock intersect(BasicBlock left, BasicBlock right,
    const BlockIndexMap& positions, const BasicBlockMap& dominatorTree)
{
    while(left != right)
    {
        auto leftPosition = positions.find(left);
        assert(leftPosition != positions.end());

        auto rightPosition = positions.find(right);
        assert(rightPosition != positions.end());

        if(leftPosition->second < rightPosition->second)
        {
            auto leftDominator = dominatorTree.find(left);
            assert(leftDominator != dominatorTree.end());

            left = leftDominator->second;
        }
        else if(rightPosition->second < leftPosition->second)
        {
            auto rightDominator = dominatorTree.find(right);
            assert(rightDominator != dominatorTree.end());

            right = rightDominator->second;
        }
    }

    return left;
}

} // namespace analysis
} // namespace lucius





