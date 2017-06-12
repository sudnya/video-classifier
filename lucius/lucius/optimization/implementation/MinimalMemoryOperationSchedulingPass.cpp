/*  \file   MinimalMemoryOperationSchedulingPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the MinimalMemoryOperationSchedulingPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/MinimalMemoryOperationSchedulingPass.h>

namespace lucius
{
namespace optimization
{

MinimalMemoryOperationSchedulingPass::MinimalMemoryOperationSchedulingPass()
{

}

MinimalMemoryOperationSchedulingPass::~MinimalMemoryOperationSchedulingPass()
{

}

static bool getInputMemory(Operation* operation)
{
    auto& operands = operation->getOperands();

    size_t totalMemory = 0;

    for(auto* operand : operands)
    {
        totalMemory += getMemoryUsage(totalMemory);
    }

    return totalMemory;
}

static Operation* getBestOperation(OperationVector& readyOperations)
{
    // sort by used memory in operands
    std::sort(readyOperations.begin(), readyOperations.end(),
        [](Operation* left, Operation* right)
        {
            return getInputMemory(left) < getInputMemory(right);
        });

    auto* bestOperation = readyOperations.back();

    readyOperations.pop_back();

    return bestOperation;
}

static void scheduleBasicBlock(OperationList& newOrder, OperationList& originalOrder)
{
    // schedule according to memory priority using a greedy algorithm
    OperationVector readyOperations;
    OperationSet    scheduledOperations;

    for(auto* operation : originalOrder)
    {
        if(isReady(operation, scheduledOperations))
        {
            readyOperations.push_back(operation);
        }
    }

    while(!readyOperations.empty())
    {
        auto* operation = getBestOperation(readyOperations);

        newOrder.push_back(operation);
        scheduledOperations.insert(operation);

        auto successors = getSuccessors(operation);

        for(auto* successor : successors)
        {
            if(isReady(successor, scheduledOperations))
            {
                readyOperations.push_back(operation);
            }
        }
    }

    newOrder.push_back(originalOrder.back());
}

void MinimalMemoryOperationSchedulingPass::runOnFunction(ir::Function& function)
{
    // schedule each basic block independently
    for(auto& basicBlock : function)
    {
        OperationList newOrder;

        scheduleBasicBlock(newOrder, basicBlock->getOperations());

        basicBlock->moveOperations(newOrder);
    }
}

StringSet MinimalMemoryOperationSchedulingPass::getRequiredAnalyses() const
{
    return StringSet();
}

} // namespace optimization
} // namespace lucius






