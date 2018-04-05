/*  \file   PostDominatorAnalysis.cpp
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The source file for the PostDominatorAnalysis class.
*/

// Lucius Includes
#include <lucius/analysis/interface/PostDominatorAnalysis.h>

#include <lucius/analysis/implementation/DominatorHelpers.h>

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
    using BasicBlockMap = std::map<BasicBlock, BasicBlock>;

    void buildPostDominatorTree(const Function& function)
    {
        buildDominatorTree(_postDominatorTree, _reversePostOrderPositions, function, false);
    }

public:
    BasicBlock getPostDominator(BasicBlock left, BasicBlock right)
    {
        return intersect(left, right, _reversePostOrderPositions, _postDominatorTree);
    }

private:
    BasicBlockMap _postDominatorTree;
    BlockIndexMap _reversePostOrderPositions;

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
    util::log("DominatorAnalysis") << " getting post-dominator of blocks (" << one.name()
        << ", " << two.name() << ")\n";

    auto result = _implementation->getPostDominator(one, two);

    util::log("DominatorAnalysis") << "  post-dominator is " << result.name() << "\n";

    return result;
}


} // namespace analysis
} // namespace lucius




