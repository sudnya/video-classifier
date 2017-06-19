/*  \file   DynamicMemoryAllocationPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the DynamicMemoryAllocatin class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimization/interface/DynamicMemoryAllocationPass.h>

namespace lucius
{
namespace optimization
{

DynamicMemoryAllocationPass::DynamicMemoryAllocationPass()
: Pass("DynamicMemoryAllocationPass")
{
    // intentionally blank
}

DynamicMemoryAllocationPass::~DynamicMemoryAllocationPass()
{
    // intentionally blank
}

static bool needsAllocation(const Operation* op)
{
    return op->getType()->isTensor();
}

static Operation* addAllocation(Operation* op)
{
    auto* target = dynamic_cast<TargetOperation*>(op);

    assert(target != nullptr);

    auto* allocation = target->getFunction()->addOperation(
        std::make_unique<AllocationOperation>(target->getType()));

    insertBefore(target, allocation);

    target->setOutputOperand(allocation);

    return allocation;
}

static void addFree(Operation* allocation, Operation* insertionPoint)
{
    auto* free = allocation->getFunction()->addOperation(
        std::make_unique<FreeOperation>(allocation));

    insertAfter(insertionPoint, free);
}

void DynamicMemoryAllocationPass::runOnFunction(ir::Function& function)
{
    auto* postDominatorAnalysis = getAnalysis<DominatorAnalysis*>("PostDominatorAnalysis");

    for(auto* basicBlock : function)
    {
        for(auto* operation : basicBlock)
        {
            if(needsAllocation(operation))
            {
                auto* allocation = addAllocation(operation);

                // free at post dominator of all uses
                auto* postDominator = getPostDominatorOfAllUses(operation, postDominatorAnalysis);

                auto* insertionPoint = nullptr;

                if(postDominatorContainsUse(operation, postDominator))
                {
                    insertionPoint = getLastUseInBlock(operation, postDominator);
                }
                else
                {
                    insertionPoint = getFirstNonPhi(postDominator);
                }

                addFree(allocation, insertionPoint);
            }
        }
    }
}

StringSet DynamicMemoryAllocationPass::getRequiredAnalyses() const
{
    return StringSet({"PostDominatorAnalysis"});
}

} // namespace optimization
} // namespace lucius







