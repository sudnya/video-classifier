/*  \file   OperationFinalizationPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the OperationFinalizationPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/OperationFinalizationPass.h>

#include <lucius/ir/interface/Function.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace optimization
{

OperationFinalizationPass::OperationFinalizationPass()
{
    // intentionally blank
}

OperationFinalizationPass::~OperationFinalizationPass()
{
    // intentionally blank
}

void OperationFinalizationPass::runOnFunction(ir::Function& )
{
    // TODO
}

StringSet OperationFinalizationPass::getRequiredAnalyses() const
{
    return StringSet();
}

Function OperationFinalizationPass::getTargetFunction()
{
    assertM(false, "Not implemented");

    return Function();
}

} // namespace optimization
} // namespace lucius





