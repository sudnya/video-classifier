/*  \file   DominatorAnalysis.cpp
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The source file for the DominatorAnalysis class.
*/

// Lucius Includes
#include <lucius/analysis/interface/DominatorAnalysis.h>

#include <lucius/ir/interface/BasicBlock.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace analysis
{

class DominatorAnalysisImplementation
{

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
    assertM(false, "Not implemented");
}

DominatorAnalysis::StringSet DominatorAnalysis::getRequiredAnalyses() const
{
    return StringSet();
}

ir::BasicBlock DominatorAnalysis::getDominator(ir::BasicBlock one, ir::BasicBlock two) const
{
    assertM(false, "Not implemented");
    return ir::BasicBlock();
}

} // namespace analysis
} // namespace lucius




