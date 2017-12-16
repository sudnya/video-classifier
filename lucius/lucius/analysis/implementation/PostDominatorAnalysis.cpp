/*  \file   PostDominatorAnalysis.cpp
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The source file for the PostDominatorAnalysis class.
*/

// Lucius Includes
#include <lucius/analysis/interface/PostDominatorAnalysis.h>

#include <lucius/analysis/interface/Traversals.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Function.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <map>

namespace lucius
{

namespace analysis
{

using BasicBlock = ir::BasicBlock;
using Function = ir::Function;

class PostDominatorAnalysisImplementation
{
public:
    using BlockIndexMap = std::map<BasicBlock, size_t>;

    void buildPostDominatorTree(const Function& function)
    {
        _postDominatorTree.clear();

        if(function.empty())
        {
            return;
        }

        auto reversePostOrder = reversePostOrderTraversal(
            BasicBlockList(function.begin(), function.end()), true);

        BlockIndexMap reversePostOrderPositions;

        for(auto& block : reversePostOrder)
        {
            reversePostOrderPositions[block] = reversePostOrderPositions.size();
        }

        bool changed = true;

        auto exitBlock = reversePostOrder.front();
        reversePostOrder.pop_front();

        _postDominatorTree[exitBlock] = exitBlock;

        while(changed)
        {
            changed = false;

            for(auto& block : reversePostOrder)
            {
                // perform post-dominator set intersections

                // find the first processed predecessor
                assert(!block.getSuccessors().empty());

                BasicBlock newImmediatePostDominator = *block.getSuccessors().begin();

                assert(_postDominatorTree.count(newImmediatePostDominator) != 0);

                // intersect
                for(auto& successor : block.getSuccessors())
                {
                    newImmediatePostDominator = _intersect(successor, newImmediatePostDominator,
                        reversePostOrderPositions);
                }

                auto postDominator = _postDominatorTree.find(block);

                if(postDominator == _postDominatorTree.end())
                {
                    _postDominatorTree.insert(std::make_pair(block, newImmediatePostDominator));
                    changed = true;
                }
                else if(postDominator->second != newImmediatePostDominator)
                {
                    postDominator->second = newImmediatePostDominator;
                    changed = true;
                }
            }
        }
    }

private:
    BasicBlock _intersect(BasicBlock left, BasicBlock right,
        const BlockIndexMap& reversePostOrderPosition)
    {
        while(left != right)
        {
            auto leftPosition = reversePostOrderPosition.find(left);
            assert(leftPosition != reversePostOrderPosition.end());

            auto rightPosition = reversePostOrderPosition.find(right);
            assert(rightPosition != reversePostOrderPosition.end());

            if(leftPosition->second < rightPosition->second)
            {
                left = _postDominatorTree[left];
            }
            else if(rightPosition->second < leftPosition->second)
            {
                right = _postDominatorTree[right];
            }
        }

        return left;
    }

private:
    std::map<BasicBlock, BasicBlock> _postDominatorTree;

};

PostDominatorAnalysis::PostDominatorAnalysis()
: _implementation(std::make_unique<PostDominatorAnalysisImplementation>())
{

}

PostDominatorAnalysis::~PostDominatorAnalysis()
{

}

void PostDominatorAnalysis::runOnFunction(const Function& function)
{
    util::log("PostDominatorAnalysis") << "Running on " << function.name() << "\n";

    _implementation->buildPostDominatorTree(function);
}

PostDominatorAnalysis::StringSet PostDominatorAnalysis::getRequiredAnalyses() const
{
    return StringSet();
}

ir::BasicBlock PostDominatorAnalysis::getPostDominator(ir::BasicBlock one,
    ir::BasicBlock two) const
{
    assertM(false, "Not implemented");
    return ir::BasicBlock();
}


} // namespace analysis
} // namespace lucius




