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

/*! \brief A class representing an optimization pass. */
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



