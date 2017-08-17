/*  \file   OperationFinalizationPass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the OperationFinalizationPass class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimization/interface/Pass.h>

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlock; } }

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
    using BasicBlockList = std::list<ir::BasicBlock>;

public:
    /* \brief Extracts the target function.  Caller takes ownership. */
    Function getTargetFunction();

public:
    void runOnFunction(ir::Function& ) final;

public:
    StringSet getRequiredAnalyses() const final;
};

} // namespace optimization
} // namespace lucius






