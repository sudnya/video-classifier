/*  \file   MemoryEfficientBackPropagationPass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the MemoryEfficientBackPropagationPass class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimization/interface/Pass.h>

namespace lucius
{
namespace optimization
{

/*! \brief An optimization pass that generates the back propagation graph using recomputation
           to avoid exceeding a memory limit.

    The current algorithm is based on linearizing the schedule, and then using exponentially
    spaced checkpoints to avoid recomputation.

    Improvement 1: Add a local search heuristic to adjust the checkpoints to further
        reduce compute requirements.

    Improvement 2: Relax the linearized schedule assumption.

    Improvement 3: Compute a globally optimal schedule using ILP and a relaxation method.

*/
class MemoryEfficientBackPropagationPass : public Pass
{
public:
    MemoryEfficientBackPropagationPass();
    virtual ~MemoryEfficientBackPropagationPass();

public:
    void runOnFunction(ir::Function& ) final;

public:
    StringSet getRequiredAnalyses() const final;
};

} // namespace optimization
} // namespace lucius



