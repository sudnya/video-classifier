/*  \file   OperationFinalizationPass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the OperationFinalizationPass class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimization/interface/Pass.h>

namespace lucius
{
namespace optimization
{

/*! \brief A class that prepares a function for execution. */
class OperationFinalizationPass : public Pass
{
public:
    OperationFinalizationPass();
    virtual ~OperationFinalizationPass();

public:
    void runOnFunction(ir::Function& ) final;

public:
    StringSet getRequiredAnalyses() const final;
};

} // namespace optimization
} // namespace lucius






