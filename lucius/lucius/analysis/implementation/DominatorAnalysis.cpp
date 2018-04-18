/*  \file   DominatorAnalysis.cpp
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The source file for the DominatorAnalysis class.
*/

// Lucius Includes
#include <lucius/analysis/interface/DominatorAnalysis.h>

#include <lucius/analysis/implementation/DominatorHelpers.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Function.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace analysis
{

class DominatorAnalysisImplementation
{
public:
    using BlockIndexMap = std::map<BasicBlock, size_t>;
    using BlockMap = std::map<BasicBlock, BasicBlock>;
    using BlockVector = std::vector<BasicBlock>;
    using BlockVectorMap = std::map<BasicBlock, BlockVector>;

    void buildForwardDominatorTree(const Function& function)
    {
        buildDominatorTree(_dominatorTree, _reversePostOrderPositions, function, true);
        buildDominanceFrontiers(_dominanceFrontiers, _dominatorTree, true);
    }

    BasicBlock getDominator(BasicBlock left, BasicBlock right) const
    {
        // find the first block that dominates both
        return intersect(left, right, _reversePostOrderPositions, _dominatorTree);
    }

    BasicBlock getImmediateDominator(BasicBlock one) const
    {
        auto dominator = _dominatorTree.find(one);

        assert(dominator != _dominatorTree.end());

        return dominator->second;
    }

    bool isDominator(BasicBlock one, BasicBlock two) const
    {
        return getDominator(one, two) == one;
    }

    BlockVector getDominanceFrontier(ir::BasicBlock block) const
    {
        auto frontier = _dominanceFrontiers.find(block);

        assert(frontier != _dominanceFrontiers.end());

        return frontier->second;
    }

    Function getFunction() const
    {
        assert(!_dominatorTree.empty());

        return _dominatorTree.begin()->first.getFunction();
    }

private:
    BlockMap       _dominatorTree;
    BlockIndexMap  _reversePostOrderPositions;
    BlockVectorMap _dominanceFrontiers;

};

DominatorAnalysis::DominatorAnalysis()
: _implementation(std::make_unique<DominatorAnalysisImplementation>())
{

}

DominatorAnalysis::~DominatorAnalysis()
{

}

void DominatorAnalysis::runOnFunction(const Function& function)
{
    util::log("DominatorAnalysis") << "Running on " << function.name() << "\n";

    _implementation->buildForwardDominatorTree(function);
}

DominatorAnalysis::StringSet DominatorAnalysis::getRequiredAnalyses() const
{
    return StringSet();
}

ir::BasicBlock DominatorAnalysis::getDominator(ir::BasicBlock one, ir::BasicBlock two) const
{
    util::log("DominatorAnalysis") << " getting dominator of blocks (" << one.name()
        << ", " << two.name() << ")\n";

    auto result = _implementation->getDominator(one, two);

    util::log("DominatorAnalysis") << "  dominator is " << result.name() << "\n";

    return result;
}

ir::BasicBlock DominatorAnalysis::getImmediateDominator(ir::BasicBlock one) const
{
    return _implementation->getImmediateDominator(one);
}

bool DominatorAnalysis::isDominator(ir::BasicBlock one, ir::BasicBlock two) const
{
    return _implementation->isDominator(one, two);
}

BlockVector DominatorAnalysis::getDominanceFrontier(ir::BasicBlock block) const
{
    return _implementation->getDominanceFrontier(block);
}

DominatorAnalysis::DominatorTree DominatorAnalysis::getDominatorTree() const
{
    auto function = _implementation->getFunction();

    DominatorTree tree;

    for(auto& block : function)
    {
        tree[block] = BlockVector();
    }

    for(auto& block : function)
    {
        auto dominator = getImmediateDominator(block);

        if(dominator == block)
        {
            continue;
        }

        tree[dominator].push_back(block);
    }

    return tree;
}

} // namespace analysis
} // namespace lucius




