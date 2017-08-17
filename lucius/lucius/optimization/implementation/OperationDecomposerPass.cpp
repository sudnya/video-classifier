/*  \file   OperationDecomposerPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the OperationDecomposerPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/OperationDecomposerPass.h>

#include <lucius/analysis/interface/OperationPerformanceAnalysis.h>

#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Function.h>

#include <lucius/matrix/interface/Dimension.h>

#include <lucius/util/interface/debug.h>

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

using Dimension = matrix::Dimension;
using Operation = ir::Operation;
using BasicBlock = ir::BasicBlock;
using OperationPerformanceAnalysis = analysis::OperationPerformanceAnalysis;

static bool isSplittingLegal(const Operation& operation)
{
    // TODO: Add more checks

    return true;
}

static bool shouldOperationBeSplit(const Operation& operation,
    const OperationPerformanceAnalysis& performanceAnalysis)
{
    // is splitting legal for the op
    if(!isSplittingLegal(operation))
    {
        return false;
    }

    // is the operation latency or throughput bound
    double time         = performanceAnalysis.getOperationTime(operation);
    double overheadTime = performanceAnalysis.getOverheadTime(operation);

    bool isThroughputLimited = time > 2.0 * overheadTime;

    return isThroughputLimited;
}

static void copyOperation(BasicBlock& newBasicBlock, const Operation& operation)
{
    newBasicBlock.push_back(operation.clone());
}

/* TODO
static bool isGemmComputeBoundForTileSize(const Operation& gemm,
    size_t tileSize, const Dimension& size,
    const OperationPerformanceAnalysis& performanceAnalysis)
{
    auto c = ir::value_cast<TensorValue>(gemm.getOperand(0));
    auto a = ir::value_cast<TensorValue>(gemm.getOperand(1));

    size_t m = std::min(c.getMaxSize()[0], tileSize);
    size_t n = std::min(c.getMaxSize()[1], tileSize);
    size_t k = std::min(a.getMaxSize()[1], tileSize);

    TensorValue cTile({m, n}, gemm.getPrecision());
    TensorValue aTile({m, k}, gemm.getPrecision());
    TensorValue bTile({k, n}, gemm.getPrecision());

    GemmOperation dummyOperation(cTile, aTile, bTile);

    return shouldOperationBeSplit(dummyOperation, performanceAnalysis);
    return true;
}
*/

static bool isGemm(const Operation& operation)
{
    // TODO

    return false;
}

static void splitGemm(BasicBlock& newBasicBlock, const Operation& operation,
    const OperationPerformanceAnalysis& performanceAnalysis)
{
    // TODO
    /*
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

    for(size_t rowSplit = 0; rowSplit < rowSplits; ++rowSplit)
    {
        for(size_t columnSplit = 0; columnsSplit < columnSplits; ++columnSplit)
        {
            // slice the inputs

            // gemm

        }
    }

    // merge the outputs
    */
}

static bool isConvolution(const Operation& operation)
{
    // TODO

    return false;
}

static void splitConvolution(BasicBlock& newBasicBlock, const Operation& operation,
    const OperationPerformanceAnalysis& performanceAnalysis)
{
    // TODO
}

static bool isPointWise(const Operation& operation)
{
    // TODO

    return false;
}

static void splitPointWise(BasicBlock& newBasicBlock, const Operation& operation,
    const OperationPerformanceAnalysis& performanceAnalysis)
{
    // TODO
}

static bool isBroadcast(const Operation& operation)
{
    // TODO

    return false;
}

static void splitBroadcast(BasicBlock& newBasicBlock, const Operation& operation,
    const OperationPerformanceAnalysis& performanceAnalysis)
{
    // TODO
}

static bool isReduce(const Operation& operation)
{
    // TODO

    return false;
}

static void splitReduce(BasicBlock& newBasicBlock, const Operation& operation,
    const OperationPerformanceAnalysis& performanceAnalysis)
{
    // TODO
}

static void splitOperation(BasicBlock& newBasicBlock, const Operation& operation,
    const OperationPerformanceAnalysis& performanceAnalysis)
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

using Function = ir::Function;

static void replaceBasicBlock(BasicBlock& output, const BasicBlock& input)
{
    //auto& function = input.getFunction();

    // TODO
}

void OperationDecomposerPass::runOnFunction(Function& function)
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
            if(!shouldOperationBeSplit(operation, *performanceAnalysis))
            {
                copyOperation(newBasicBlock, operation);
                continue;
            }

            splitOperation(newBasicBlock, operation, *performanceAnalysis);
        }

        replaceBasicBlock(newBasicBlock, basicBlock);
    }
}

StringSet OperationDecomposerPass::getRequiredAnalyses() const
{
    return StringSet({"OperationPerformanceAnalysis"});
}

} // namespace optimization
} // namespace lucius

