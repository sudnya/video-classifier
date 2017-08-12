/*  \file   DominatorAnalysis.h
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The header file for the DominatorAnalysis class.
*/

#pragma once

// Lucius Includes
#include <lucius/analysis/interface/Analysis.h>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlock; } }

namespace lucius
{

namespace analysis
{

/*! \brief A class for representing an analysis. */
class DominatorAnalysis : public Analysis
{
public:
    DominatorAnalysis();
    virtual ~DominatorAnalysis() final;

public:
    virtual void runOnFunction(const Function& function) final;

public:
    virtual StringSet getRequiredAnalyses() const final;

public:
    ir::BasicBlock getDominator(ir::BasicBlock one, ir::BasicBlock two) const;
};

} // namespace analysis
} // namespace lucius



