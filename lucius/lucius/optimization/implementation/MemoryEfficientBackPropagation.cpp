/*  \file   MemoryEfficientBackPropagationPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the MemoryEfficientBackPropagationPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/MemoryEfficientBackPropagationPass.cpp>

namespace lucius
{
namespace optimization
{

MemoryEfficientBackPropagationPass::MemoryEfficientBackPropagationPass()
{
    // intentionally blank
}

MemoryEfficientBackPropagationPass::~MemoryEfficientBackPropagationPass()
{
    // intentionally blank
}

static double getAvailableMemoryForBackPropagation()
{
    double totalSystemMemory = machine::MachineModel::getTotalSystemMemoryCapacity();

    // TODO: reserve memory for overheads

    return totalSystemMemory;
}

static OperationVectors partitionGraphIntoWavefronts(ir::Function& function)
{
    OperationVectors wavefronts;

    auto readyOperations = getReadyOpertations(function);

    OperationSet finishedOperations;

    while(!readyOperations.empty())
    {
        OperationSet nextReadyOperations;

        wavefront.push_back(OperationVector(readyOperations.begin(), readyOperations.end()));

        finishedOperations.insert(readyOperations.begin(), readyOperations.end());

        for(auto* operation : readyOperation)
        {
            auto successors = getSuccessors(operation);

            for(auto* successor : successors)
            {
                if(isReady(successor, finishedOperations))
                {
                    nextReadyOperations.insert(successor);
                }
            }
        }

        readyOperations = std::move(nextReadyOperations);
    }

    return wavefronts;
}

double getForwardAndBackPropWorkingSpaceRequirement(const OperationVectors& wavefronts,
    OperationMemoryAnalysis* memoryAnalysis)
{
    double workingSpaceSize = 0.0;

    // TODO: This naive version reserves the entire working set of the wavefront, rather
    //       than attempting intelligent scheduling
    for(auto& wavefront : wavefronts)
    {
        double wavefrontWorkingSpaceSize = 0.0;

        for(auto* operation : wavefront)
        {
            wavefrontWorkingSpaceSize += memoryAnalysis->getOperationMemoryRequirement(operation);
        }

        workingSpaceSize = std::max(workingSpaceSize, wavefrontWorkingSpaceSize);
    }

    return workingSpaceSize;
}

static ValueSets schedulePhases(const OperationVectors& wavefronts, double availableMemory)
{
    IntVectors schedule;

    IntVector remainingWavefronts = range(wavefronts.size());

    while(!remainingWavefronts.empty())
    {
        double phaseAvailableMemory = availableMemory;

        IntVector currentSchedule;

        // schedule values in the last wavefront
        currentSchedule.push_back(remainingWavefronts.back());

        decrementAvailableMemory(phaseAvailableMemory, wavefronts, remainingWavefronts.back());

        // recursively greedy schedule
        bisectSchedule(currentSchedule, remainingWavefronts, wavefronts, phaseAvailableMemory);
    }

    ValueSets values;

    for(auto& phase : schedule)
    {
        ValueSet valuesSavedThisPhase;

        for(auto& wavefrontId : phase)
        {
            auto& wavefront = wavefronts[wavefrontId];

            for(auto* operation : wavefront)
            {
                auto values = operation->getAllValues();

                for(auto* value : values)
                {
                    valuesSavedThisPhase.insert(value);
                }
            }
        }

        values.emplace_back(std::move(valuesSavedThisPhase));
    }

    return values;
}

static void generateOperations(ir::Function& function, const ValueSets& phases)
{
    BasicBlockList newBasicBlocks;

    ValueSet savedValues;

    for(auto& phase : phases)
    {
        // TODO
    }

    replaceBasicBlocks(function, newBasicBlocks);
}

void MemoryEfficientBackPropagationPass::runOnFunction(ir::Function& function)
{
    auto* memoryAnalysis = dynamic_cast<OperationMemoryAnalysis*>(
        getAnalysis("OperationMemoryAnalysis"));

    // get memory available for back prop
    double availableMemory = getAvailableMemoryForBackPropagation();

    // 1. partition the graph into wavefronts along the critical path
    auto wavefronts = partitionGraphIntoWavefronts(function);

    // 2. reserve the working set for forward prop and back prop
    double workingSpace = getForwardAndBackPropWorkingSpaceRequirement(wavefronts);

    availableMemory = std::max(0.0, availableMemory - workingSpace);

    // 3. divide the back propagation operation into phases
    auto phases = schedulePhases(wavefronts, availableMemory);

    // 4. generate operations for each phase
    generateOperations(function, phases);
}

StringSet MemoryEfficientBackPropagationPass::getRequiredAnalyses() const
{
    return StringSet({"OperationMemoryAnalysis"});
}

} // namespace optimization
} // namespace lucius




