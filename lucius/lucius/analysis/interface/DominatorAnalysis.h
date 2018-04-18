/*  \file   DominatorAnalysis.h
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The header file for the DominatorAnalysis class.
*/

#pragma once

// Lucius Includes
#include <lucius/analysis/interface/Analysis.h>

// Standard Library Includes
#include <vector>
#include <map>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlock; } }

namespace lucius { namespace analysis { class DominatorAnalysisImplementation; } }

namespace lucius
{

namespace analysis
{

/*! \brief A class for computing dominator trees and dominance frontiers.

    Currently uses: https://www.cs.rice.edu/~keith/EMBED/dom.pdf

    TODO: Add incremental version ( https://arxiv.org/abs/1604.02711 )
*/
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
    using BlockVector = std::vector<ir::BasicBlock>;
    using DominatorTree = std::map<ir::BasicBlock, BlockVector>;

public:
    /*! \brief Get the first block that dominates one and two. */
    ir::BasicBlock getDominator(ir::BasicBlock one, ir::BasicBlock two) const;

    /*! \brief Get the immediate dominator of one. */
    ir::BasicBlock getImmediateDominator(ir::BasicBlock one) const;

    /*! \brief Does one dominate two? */
    bool isDominator(ir::BasicBlock one, ir::BasicBlock two) const;

    /*! \brief Get the set of blocks for which the selected block's dominance ends. */
    BlockVector getDominanceFrontier(ir::BasicBlock one) const;

    /*! \brief Get the full dominator tree. */
    DominatorTree getDominatorTree() const;

private:
    std::unique_ptr<DominatorAnalysisImplementation> _implementation;
};

} // namespace analysis
} // namespace lucius



