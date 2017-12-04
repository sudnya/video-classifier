/*  \file   PostDominatorAnalysis.h
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The header file for the PostDominatorAnalysis class.
*/

#pragma once

// Lucius Includes
#include <lucius/analysis/interface/Analysis.h>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlock; } }

namespace lucius { namespace analysis { class PostDominatorAnalysisImplementation; } }

namespace lucius
{

namespace analysis
{

/*! \brief A class for representing an analysis. */
class PostDominatorAnalysis : public Analysis
{
public:
    PostDominatorAnalysis();
    virtual ~PostDominatorAnalysis() final;

public:
    virtual void runOnFunction(const Function& function) final;

public:
    virtual StringSet getRequiredAnalyses() const final;

public:
    ir::BasicBlock getPostDominator(ir::BasicBlock one, ir::BasicBlock two) const;

private:
    std::unique_ptr<PostDominatorAnalysisImplementation> _implementation;
};

} // namespace analysis
} // namespace lucius



