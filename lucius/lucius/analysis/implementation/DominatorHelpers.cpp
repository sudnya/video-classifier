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

// Standard Library Includes
#include <set>

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
        positions[block] = traversal.size() - positions.size() - 1;
    }

    bool changed = true;

    auto startBlock = traversal.front();
    traversal.pop_front();

    tree[startBlock] = startBlock;

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

            auto possibleBlock = nextBlocks.begin();
            bool anyProcessed = false;

            for( ; possibleBlock != nextBlocks.end(); ++possibleBlock)
            {
                anyProcessed = tree.count(*possibleBlock) != 0;

                if(anyProcessed)
                {
                    break;
                }
            }

            if(!anyProcessed)
            {
                continue;
            }

            auto newImmediateDominator = *possibleBlock;

            // intersect
            for(auto& nextBlock : nextBlocks)
            {
                if(tree.count(nextBlock) == 0)
                {
                    continue;
                }

                newImmediateDominator = intersect(nextBlock, newImmediateDominator,
                    positions, tree);
            }

            // update tree
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

void buildDominanceFrontiers(BlockVectorMap& dominanceFrontiers,
    const BasicBlockMap& dominatorTree, bool isDominatorAnalysis)
{
    std::map<BasicBlock, std::set<BasicBlock>> frontiers;

    for(auto& entry : dominatorTree)
    {
        auto& block = entry.first;
        auto& dominator = entry.second;

        dominanceFrontiers.insert(std::make_pair(block, BlockVector()));

        auto predecessors = isDominatorAnalysis ? block.getPredecessors() : block.getSuccessors();

        if(predecessors.size() < 2)
        {
            continue;
        }

        for(auto& predecessor : predecessors)
        {
            auto runner = predecessor;

            while(runner != dominator)
            {
                frontiers[runner].insert(block);

                auto newRunner = dominatorTree.find(runner);

                assert(newRunner != dominatorTree.end());

                runner = newRunner->second;
            }
        }
    }

    for(auto& entry : frontiers)
    {
        dominanceFrontiers[entry.first] = BlockVector(entry.second.begin(), entry.second.end());
    }
}

} // namespace analysis
} // namespace lucius





