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

#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace optimization
{

MinimalMemoryOperationSchedulingPass::MinimalMemoryOperationSchedulingPass()
: Pass("MinimalMemoryOperationSchedulingPass")
{

}

MinimalMemoryOperationSchedulingPass::~MinimalMemoryOperationSchedulingPass()
{

}

using Operation = ir::Operation;
using OperationMemoryAnalysis = analysis::OperationMemoryAnalysis;

static double getInputMemory(const Operation& operation,
    const OperationMemoryAnalysis& memoryAnalysis)
{
    auto operands = operation.getOperands();

    double totalMemory = 0;

    for(auto& operand : operands)
    {
        totalMemory += memoryAnalysis.getOperandMemoryRequirement(operand);
    }

    util::log("MinimalMemoryOperationSchedulingPass") << "    Operation '"
        << operation.toString() << "' uses " << totalMemory << " bytes.\n";

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
    if(operation.isControlOperation())
    {
        return false;
    }

    auto predecessors = operation.getPredecessors();

    for(auto& predecessor : predecessors)
    {
        // skip PHIs
        if(predecessor.isPHI())
        {
            continue;
        }

        // skip predecessors in different blocks
        if(predecessor.getBasicBlock() != operation.getBasicBlock())
        {
            continue;
        }

        // predecessors in the same block should occur previously in program order
        if(predecessor.getIndexInBasicBlock() > operation.getIndexInBasicBlock())
        {
            assert(false);
            continue;
        }

        if(scheduledOperations.count(predecessor) == 0)
        {
            return false;
        }
    }

    return true;
}

static void scheduleBasicBlock(OperationList& newOrder, const OperationList& originalOrder,
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

        util::log("MinimalMemoryOperationSchedulingPass") << "   scheduling operation  '"
            << operation.toString() << "'.\n";
        newOrder.push_back(operation);
        scheduledOperations.insert(operation);

        auto successors = operation.getSuccessors();

        for(auto& successor : successors)
        {
            // skip successsors in different blocks
            if(successor.getBasicBlock() != operation.getBasicBlock())
            {
                continue;
            }

            // skip PHIs
            if(successor.isPHI())
            {
                continue;
            }

            // already scheduled ops
            assert(scheduledOperations.count(successor) == 0);

            if(isReady(successor, scheduledOperations))
            {
                readyOperations.push_back(successor);
            }
        }
    }

    // add the branch back in
    if(originalOrder.back().isControlOperation())
    {
        util::log("MinimalMemoryOperationSchedulingPass") << "   scheduling control operation  '"
            << originalOrder.back().toString() << "'.\n";
        newOrder.push_back(originalOrder.back());
    }

    assert(newOrder.size() == originalOrder.front().getBasicBlock().size());
}

void MinimalMemoryOperationSchedulingPass::runOnFunction(ir::Function& function)
{
    const OperationMemoryAnalysis* memoryAnalysis =
        static_cast<OperationMemoryAnalysis*>(getAnalysis("OperationMemoryAnalysis"));

    util::log("MinimalMemoryOperationSchedulingPass") << "  scheduling operations in function '"
        << function.name() << "'.\n";

    // schedule each basic block independently
    for(auto& basicBlock : function)
    {
        util::log("MinimalMemoryOperationSchedulingPass")
            << "  scheduling operations in basic block '"
            << basicBlock.name() << "'.\n";

        OperationList newOrder(basicBlock.phiBegin(), basicBlock.phiEnd());

        scheduleBasicBlock(newOrder, OperationList(basicBlock.phiEnd(), basicBlock.end()),
            *memoryAnalysis);

        basicBlock.setOperations(std::move(newOrder));
    }
    util::log("MinimalMemoryOperationSchedulingPass") << "  function is now '"
        << function.toString() << "'.\n";
}

StringSet MinimalMemoryOperationSchedulingPass::getRequiredAnalyses() const
{
    return StringSet();
}

} // namespace optimization
} // namespace lucius






