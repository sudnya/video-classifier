/*  \file   Traversals.cpp
    \author Gregory Diamos
    \date   December 8, 2017
    \brief  The source file for the set of IR traversal functions.
*/

// Lucius Includes
#include <lucius/analysis/interface/Traversals.h>

#include <lucius/ir/interface/BasicBlock.h>


// Standard Library Includes
#include <set>
#include <stack>

namespace lucius
{

namespace analysis
{

using BasicBlock = ir::BasicBlock;
using BasicBlockList = std::list<ir::BasicBlock>;

BasicBlockList reversePostOrderTraversal(const BasicBlockList& blocks, bool reverseEdges)
{
    auto postOrder = postOrderTraversal(blocks, reverseEdges);

    std::reverse(postOrder.begin(), postOrder.end());

    return postOrder;
}

using BasicBlockStack = std::stack<std::pair<BasicBlock, bool>>;
using BasicBlockSet = std::set<BasicBlock>;
using BasicBlockList = std::list<BasicBlock>;

static void tryStackPush(BasicBlockStack& stack, const BasicBlock& block,
    BasicBlockSet& visited, const BasicBlockSet& reachable)
{
    if(reachable.count(block) == 0)
    {
        return;
    }

    auto visit = visited.insert(block);

    if(!visit.second)
    {
        return;
    }

    stack.push(std::make_pair(block, false));
}

BasicBlockList postOrderTraversal(const BasicBlockList& blocks, bool reverseEdges)
{
    BasicBlockSet reachable(blocks.begin(), blocks.end());
    BasicBlockSet visited;

    BasicBlockStack stack;

    stack.push(std::make_pair(blocks.front(), false));
    visited.insert(blocks.front());

    BasicBlockList order;

    while(!stack.empty())
    {
        auto& next = stack.top();

        // add successors if not already traversed
        if(!next.second)
        {
            if(!reverseEdges)
            {
                for(auto& successor : next.first.getSuccessors())
                {
                    tryStackPush(stack, successor, visited, reachable);
                }
            }
            else
            {
                for(auto& predecessor : next.first.getPredecessors())
                {
                    tryStackPush(stack, predecessor, visited, reachable);
                }
            }

            next.second = true;
        }
        else
        {
            order.push_back(next.first);
            stack.pop();
        }
    }

    return order;
}

} // namespace analysis
} // namespace lucius




