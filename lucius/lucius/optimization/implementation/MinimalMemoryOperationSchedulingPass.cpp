/*  \file   MinimalMemoryOperationSchedulingPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the MinimalMemoryOperationSchedulingPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/MinimalMemoryOperationSchedulingPass.h>

#include <lucius/analysis/interface/OperationMemoryAnalysis.h>

#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/BasicBlock.h>

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

using Operation = ir::Operation;
using OperationMemoryAnalysis = analysis::OperationMemoryAnalysis;

static bool getInputMemory(const Operation& operation,
    const OperationMemoryAnalysis& memoryAnalysis)
{
    auto operands = operation.getOperands();

    double totalMemory = 0;

    for(auto& operand : operands)
    {
        totalMemory += memoryAnalysis.getOperandMemoryRequirement(operand);
    }

    return totalMemory;
}

using OperationVector = std::vector<Operation>;

static Operation getBestOperation(OperationVector& readyOperations,
    const OperationMemoryAnalysis& memoryAnalysis)
{
    // sort by used memory in operands, the idea here is to prefer freeing up memory from
    // the most memory intensive operations first
    std::sort(readyOperations.begin(), readyOperations.end(),
        [&](const Operation& left, const Operation& right)
        {
            return getInputMemory(left, memoryAnalysis) < getInputMemory(right, memoryAnalysis);
        });

    auto bestOperation = readyOperations.back();

    readyOperations.pop_back();

    return bestOperation;
}

using OperationList = std::list<Operation>;
using OperationSet  = std::set<Operation>;

static bool isReady(const Operation& operation, const OperationSet& scheduledOperations)
{
    auto predecessors = operation.getPredecessors();

    for(auto& predecessor : predecessors)
    {
        if(scheduledOperations.count(predecessor) == 0)
        {
            return false;
        }
    }

    return true;
}

static void scheduleBasicBlock(OperationList& newOrder, OperationList& originalOrder,
    const OperationMemoryAnalysis& memoryAnalysis)
{
    // schedule according to memory priority using a greedy algorithm
    OperationVector readyOperations;
    OperationSet    scheduledOperations;

    for(auto& operation : originalOrder)
    {
        if(isReady(operation, scheduledOperations))
        {
            readyOperations.push_back(operation);
        }
    }

    while(!readyOperations.empty())
    {
        auto operation = getBestOperation(readyOperations, memoryAnalysis);

        newOrder.push_back(operation);
        scheduledOperations.insert(operation);

        auto successors = operation.getSuccessors();

        for(auto& successor : successors)
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
    const OperationMemoryAnalysis* memoryAnalysis =
        static_cast<OperationMemoryAnalysis*>(getAnalysis("OperationMemoryAnalysis"));

    // schedule each basic block independently
    for(auto& basicBlock : function)
    {
        OperationList newOrder;

        scheduleBasicBlock(newOrder, basicBlock.getOperations(), *memoryAnalysis);

        basicBlock.setOperations(std::move(newOrder));
    }
}

StringSet MinimalMemoryOperationSchedulingPass::getRequiredAnalyses() const
{
    return StringSet();
}

} // namespace optimization
} // namespace lucius






