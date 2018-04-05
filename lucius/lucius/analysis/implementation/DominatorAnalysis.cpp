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

    void buildForwardDominatorTree(const Function& function)
    {
        buildDominatorTree(_dominatorTree, _reversePostOrderPositions, function, true);
    }

    BasicBlock getDominator(BasicBlock left, BasicBlock right) const
    {
        // find the first block that dominates both
        return intersect(left, right, _reversePostOrderPositions, _dominatorTree);
    }

private:
    BlockMap      _dominatorTree;
    BlockIndexMap _reversePostOrderPositions;

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

} // namespace analysis
} // namespace lucius




