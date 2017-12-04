/*  \file   PostDominatorAnalysis.cpp
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The source file for the PostDominatorAnalysis class.
*/

// Lucius Includes
#include <lucius/analysis/interface/PostDominatorAnalysis.h>

#include <lucius/ir/interface/BasicBlock.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace analysis
{

class PostDominatorAnalysisImplementation
{

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
    assertM(false, "Not implemented");
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




