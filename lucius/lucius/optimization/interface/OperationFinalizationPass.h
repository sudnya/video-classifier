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

namespace lucius { namespace optimization { class OperationFinalizationPassImplementation; } }

namespace lucius
{
namespace optimization
{

/*! \brief A class that prepares a function for execution.


    Convert out of SSA.
*/
class OperationFinalizationPass : public FunctionPass
{
public:
    OperationFinalizationPass();
    virtual ~OperationFinalizationPass();

public:
    using BasicBlockList = std::list<ir::BasicBlock>;

public:
    /* \brief Extracts the target function. */
    Function getTargetFunction();

public:
    void runOnFunction(ir::Function& ) final;

public:
    StringSet getRequiredAnalyses() const final;

private:
    std::unique_ptr<OperationFinalizationPassImplementation> _implementation;
};

} // namespace optimization
} // namespace lucius






