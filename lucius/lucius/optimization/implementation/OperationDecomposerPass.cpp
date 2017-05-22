/*  \file   OperationDecomposerPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the OperationDecomposerPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/OperationDecomposerPass.h>

namespace lucius
{
namespace optimization
{

OperationDecomposerPass::OperationDecomposerPass()
{

}

OperationDecomposerPass::~OperationDecomposerPass()
{
    // intentionally blank
}

static bool isSplittingLegal(Operation* operation)
{
    // TODO: Add more checks

    return true;
}

static bool shouldOperationBeSplit(Operation* operation,
    OperationPerformanceAnalysis* performanceAnalysis)
{
    // is splitting legal for the op
    if(!isSplittingLegal(operation))
    {
        return false;
    }

    // is the operation latency or throughput bound
    double time         = performanceAnalysis->getOperationTime(operation);
    double overheadTime = performanceAnalysis->getOverheadTime(operation);

    bool isThroughputLimited = time > 2.0 * overheadTime;

    return isThroughputLimited;
}

static void moveOperation(BasicBlock* newBasicBlock, Operation* operation)
{
    newBasicBlock->push_back(operation->clone());
}

static void splitOperation(BasicBlock* newBasicBlock, Operation* operation,
    OperationPerformanceAnalysis* performanceAnalysis)
{
    // determine the dimension to split along
    // TODO

    // determine the number of splits

    // mechanically split the operation

    // add the new operations to the new basic block
}

static void replaceBasicBlock(BasicBlock* output, BasicBlock* input)
{
    auto* function = input->getParent();

    function->replaceBasicBlock(output, input);
}

void OperationDecomposerPass::runOnFunction(ir::Function& function)
{
    auto* performanceAnalysis = dynamic_cast<OperationPerformanceAnalysis*>(
        getAnalysis("OperationPerformanceAnalysis"));

    // run on each operation, decide whether or not to crack it based on
    // the smallest compute limited unit
    for(auto& basicBlock : function)
    {
        BasicBlock newBasicBlock;

        for(auto& operation : basicBlock)
        {
            if(!shouldOperationBeSplit(operation, performanceAnalysis))
            {
                copyOperation(newBasicBlock, operation);
                continue;
            }

            splitOperation(newBasicBlock, operation, performanceAnalysis);
        }

        replaceBasicBlock(newBasicBlock, basicBlock);
    }
}

StringSet OperationDecomposerPass::getRequiredAnalyses() const
{
    return StringSet("OperationPerformanceAnalysis");
}

} // namespace optimization
} // namespace lucius

