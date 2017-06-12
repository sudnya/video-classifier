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

static bool isGemmComputeBoundForTileSize(GemmOperation* gemm,
    size_t tileSize, const Dimension& size, OperationPerformanceAnalysis* performanceAnalysis)
{
    auto* c = dynamic_cast<TensorValue*>(gemm->getOperand(0));
    auto* a = dynamic_cast<TensorValue*>(gemm->getOperand(1));
    auto* b = dynamic_cast<TensorValue*>(gemm->getOperand(2));

    size_t m = std::min(c->getSize()[0], tileSize);
    size_t n = std::min(c->getSize()[1], tileSize);
    size_t k = std::min(a->getSize()[1], tileSize);

    TensorValue cTile({m, n}, gemm->getPrecision());
    TensorValue aTile({m, k}, gemm->getPrecision());
    TensorValue bTile({k, n}, gemm->getPrecision());

    GemmOperation dummyOperation(&cTile, &aTile, &bTile);

    return shouldOperationBeSplit(&dummyOperation, performanceAnalysis);
}

static void splitGemm(BasicBlock* newBasicBlock, Operation* operation,
    OperationPerformanceAnalysis* performanceAnalysis)
{
    auto* gemm = static_cast<GemmOperation*>(operation);

    auto* outputTensor = dynamic_cast<TensorValue*>(gemm->getOutputOperand());

    assert(outputTensor != nullptr);

    auto size = outputTensor->getSize();

    // find the minimal tile size
    size_t minimumTileSize = 1;
    size_t maximumTileSize = std::max(size[0], size[1]);

    // binary search over tile sizes
    while(minimumTileSize < maximumTileSize)
    {
        size_t midPoint = (minimumTileSize + maximumTileSize) / 2;

        if(isGemmComputeBoundForTileSize(gemm, midPoint, size, performanceAnalysis))
        {
            minimumTileSize = midPoint + 1;
        }
        else
        {
            maximumTileSize = midPoint;
        }
    }

    // do the split
    size_t tileSize = minimumTileSize;

    size_t rows    = size[0];
    size_t columns = size[1];

    size_t rowSplits    = (rows    + tileSize - 1) / tileSize;
    size_t columnSplits = (columns + tileSize - 1) / tileSize;

    // TODO
    for(size_t rowSplit = 0; rowSplit < rowSplits; ++rowSplit)
    {
        for(size_t columnSplit = 0; columnsSplit < columnSplits; ++columnSplit)
        {
            // slice the inputs

            // gemm

        }
    }

    // merge the outputs
}

static void splitOperation(BasicBlock* newBasicBlock, Operation* operation,
    OperationPerformanceAnalysis* performanceAnalysis)
{
    // mechanically split the operation

    // gemm       ops : split into squarish tiles
    // conv       ops : special rules
    // point-wise ops : split into contiguous arrays
    // broadcst   ops : split along the input
    // reduce     ops : split along the output

    if(isGemm(operation))
    {
        splitGemm(newBasicBlock, operation, performanceAnalysis);
    }
    else if(isConvolution(operation))
    {
        splitConvolution(newBasicBlock, operation, performanceAnalysis);
    }
    else if(isPointWise(operation))
    {
        splitPointWise(newBasicBlock, operation, performanceAnalysis);
    }
    else if(isBroadcast(operation))
    {
        splitBroadcast(newBasicBlock, operation, performanceAnalysis);
    }
    else if(isReduce(operation))
    {
        splitReduce(newBasicBlock, operation, performanceAnalysis);
    }
    else
    {
        assertM(false, "Invalid operation type for splitting.");
    }
}

static void replaceBasicBlock(BasicBlock* output, BasicBlock* input)
{
    auto* function = input->getFunction();

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

