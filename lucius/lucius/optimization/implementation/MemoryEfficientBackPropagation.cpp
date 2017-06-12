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

static ValueSet getGradientValues(ir::BasicBlock& basicBlock)
{
    ValueSet gradientValues;

    for(auto& operation : basicBlock)
    {
        auto* gradientOperation = dynamic_cast<GradientOperation*>(operation.get());

        if(gradientOperation == nullptr)
        {
            continue;
        }

        gradientValues.insert(gradientOperation->getOutputValue());
    }

    return gradientValue;
}

static ValueSet getGradientValues(ir::Function& function)
{
    ValueSet gradientValues;

    for(auto& block : function)
    {
        auto blockValues = getGradientValues(block);

        gradientValues.insert(blockValues.begin(), blockValues.end());
    }

    return gradientValue;
}

static BasicBlockSet getBasicBlocksWithGradientOperations(ir::Function& function)
{
    BasicBlockSet blocks;

    auto gradientValues = getGradientValues(function);

    for(auto* value : gradientValues)
    {
        auto* operation = value->getDefiningOperation();

        if(operation == nullptr)
        {
            continue;
        }

        blocks.insert(operation->getBasicBlock());
    }

    return blocks;
}

static OperationSet getOperationBackSlice(const ValueSet& gradientValues)
{
    OperationSet backSlice;

    OperationSet frontier;

    for(auto* value : gradientValues)
    {
        auto* operation = value->getDefiningOperation();

        if(operation == nullptr)
        {
            continue;
        }

        frontier.insert(operation);
    }

    while(!frontier.empty())
    {
        auto* next = frontier.begin();
        frontier.erase(frontier.begin());

        backSlice.insert(next);

        auto predecessors = next->getPredecessors();

        for(auto* predecessor : predecessors)
        {
            if(backSlice.count(predecessor) != 0)
            {
                continue;
            }

            frontier.insert(predecessor);
        }
    }

    return backSlice;
}

static OperationVectors partitionGraphIntoWavefronts(ir::Function& function,
    ir::BasicBlock& basicBlock)
{
    OperationVectors wavefronts;

    auto gradientValues = getGradientValues(basicBlock);

    auto backSlice = getOperationBackSlice(gradientValues);

    auto readyOperations = getReadyOpertationsInBackSlice(function, backSlice);

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
                if(!isReady(successor, finishedOperations))
                {
                    continue;
                }

                // ignore ops not in the back slice
                if(backSlice.count(successor) == 0)
                {
                    continue;
                }

                nextReadyOperations.insert(successor);
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

static void addBackSlice(ValueMap& availableValues,
    ValueMap& savedValues, ir::Value* value, ir::Function& function)
{
    if(savedValues.count(value) != 0)
    {
        return;
    }

    // add predecessors until we hit saved values or available values, add available values
    // along the way
    ValueSet frontier;

    frontier.insert(value);

    while(!frontier.empty())
    {
        auto* next = *frontier.begin();
        frontier.erase(frontier.begin());

        availableValues.emplace(next, next->clone());

        auto inputValues = next->getPredecessorValues();

        for(auto* inputValue : inputValues)
        {
            if(alreadyExists(inputValue, availableValues, savedValues, frontier))
            {
                continue;
            }

            frontier.insert(inputValue);
        }
    }
}

static void splitBasicBlock(InsertionPoint& insertionPoint)
{
    auto newBlock = function.insertBasicBlock(insertionPoint.getBlock());

    newBlock.insert(insertionPoint.getBlock()->begin(), insertionPoint.getOperation());
}

static void insertSubgraph(InsertionPoint& insertionPoint, BasicBlockList& subgraph)
{
    function.insert(insertionPoint, subgraph);
}

static void addRecomputeOperations(ValueMap& newlyAvailableValues, ValueMap& savedValues)
{
    BasicBlockList subgraph;
    BasicBlockMap  blockMap;

    // Create the subgraph of operations that compute available values
    for(auto* value : newlyAvailableValues)
    {
        auto* operation = value->getDefiningOperation();

        auto* block = operation->getBasicBlock();

        auto* newBlock = addBasicBlock(subgraph, blockMap, block);

        auto operands = operation->getAllValues();

        constructOperation(operation->type(), operands, newBlock);
    }

    // Insert the new subgraph
    insertSubgraph(insertionPoint, subgraph);
}

static void saveValues(ValueMap& savedValues, const ValueMap& availableValues,
    const ValueSet& phase)
{
    ValueMap newSavedValues;

    for(auto* value : phase)
    {
        auto savedValue = savedValues.find(value);

        if(savedValue != savedValues.end())
        {
            newSavedValues[value] = savedValue->second;
            continue;
        }

        auto availableValue = availableValues.find(value);

        assert(availableValue != availableValues.end());

        newSavedValues[values] = availableValue->second;
    }

    savedValues = std::move(newSavedValues);
}

static void addConnectedValues(ValueSet& frontier, Value* value,
    const ValueMap& gradientValues)
{
    auto connectedValues = getDirectlyConnectedValues(value);

    for(auto* connectedValue : connectedValues)
    {
        if(gradientValues.count(connectedValue) != 0)
        {
            continue;
        }

        if(!canGradientBeComputed(connectedValue, savedValues, gradientValues))
        {
            continue;
        }

        frontier.insert(connectedValue);
    }
}

static void addBackPropagationBranch(BasicBlock* successor, BasicBlock* newBlock)
{
    assertM(false, "Not Implemented.");
}

static BasicBlockMap::iterator setupNewBasicBlockForBackPropagation(
    BasicBlock* originalBasicBlock, BasicBlockList& subgraph, BasicBlockMap& basicBlockMap)
{
    auto newBlockIterator = subgraph.insert(subgraph.end(), BasicBlock());

    auto* newBlock = &(*newBlockIterator);

    basicBlockMap[originalBasicBlock] = newBlock;

    auto successors = originalBasicBlock->getSuccessors();

    // reverse edges, connect to successors
    for(auto* successor : successors)
    {
        auto successorMapping = basicBlockMap.find(successor);

        if(successorMapping == basicBlockMap.end())
        {
            continue;
        }

        addBackPropagationBranch(successorMapping->second, newBlock);
    }

    return newBlock;
}

static void attachOperationToSubgraph(BasicBlockList& subgraph,
    BasicBlockMap& basicBlockMap, std::unique_ptr<Operation>&& op, BasicBlock* originalBasicBlock)
{
    auto basicBlock = basicBlockMap.find(originalBasicBlock);

    if(basicBlock == basicBlockMap.end())
    {
        basicBlock = setupNewBasicBlockForBackProp(originalBasicBlock, subgraph, basicBlockMap);
    }

    basicBlock->second->push_back(std::move(op));
}

static Value* addGradientStepOperation(BasicBlockList& subgraph,
    BasicBlockMap& basicBlockMap, Value* value, const ValueMap& savedValues,
    const ValueMap& gradientValues, BasicBlock* originalBasicBlock)
{
    auto uses = value->getUses();

    Value* gradientValue = nullptr;
    auto* module = value->getModule();

    for(auto* use : uses)
    {
        auto* operation = use->getOperation();

        auto operands = setupBackPropagationOperands(operation, gradientValues);

        auto backPropOperation = OperationFactory::createBackPropOperation(operation->id());

        backPropOperation->setOperands(operands);

        auto* value = backPropOperation->getOutputValue();

        attachOperationToSubgraph(subgraph, basicBlockMap, std::move(backPropOperation),
            originalBasicBlock);

        if(gradientValue == nullptr)
        {
            gradientValue = value;
        }
        else
        {
            auto addOperation = OperationFactory::create(Add());

            addOperation->setOperands({module->addValue(value->clone()),
                gradientValue, value});

            attachOperationToSubgraph(subgraph, basicBlockMap, std::move(addOperation),
                originalBasicBlock);
        }
    }

    return gradientValue;
}

static void addBackpropagationOperations(ValueMap& gradientValues, const ValueMap& savedValues,
    const InsertionPoint& insertionPoint)
{
    ValueSet finishedValues;
    ValueSet frontier;

    BasicBlockList subgraph;
    BasicBlockMap  basicBlockMap;

    // Initialize possible gradient values that could be computed
    for(auto& valuePair : savedValues)
    {
        auto* originalValue = valuePair->first;

        addConnectedValues(frontier, originalValue, gradientValues);
    }

    // add operations to compute gradients that are ready
    while(!frontier.empty())
    {
        auto* value = *frontier.begin();
        frontier.erase(frontier.begin());

        auto* operation = value->getDefiningOperation();

        auto* basicBlock = operation->getBasicBlock();

        auto* gradient = addGradientStepOperation(subgraph, basicBlockMap, possibleValue,
            savedValues, gradientValues, basicBlock);

        gradientValues.emplace(possibleValue, gradient);

        addConnectedValues(possibleValues, possibleValue, gradientValues);
    }

    // insert basic blocks
    insertSubgraph(subgraph, insertionPoint);
}

static void generateOperations(ir::Function& function, const ValueSets& phases,
    const InsertionPoint& insertionPoint)
{
    ValueMap savedValues;
    ValueMap gradientValues;

    for(auto& phase : phases)
    {
        ValueMap availableValues;

        for(auto* value : phase)
        {
            addBackSlice(availableValues, savedValues, value, function);
        }

        addRecomputeOperations(availableValues, savedValues, insertionPoint);

        saveValues(savedValues, availableValues, phase);

        addBackpropagationOperations(gradientValues, savedValues, insertionPoint);
    }
}

void MemoryEfficientBackPropagationPass::runOnFunction(ir::Function& function)
{
    auto* memoryAnalysis = dynamic_cast<OperationMemoryAnalysis*>(
        getAnalysis("OperationMemoryAnalysis"));

    // get memory available for back prop
    double availableMemory = getAvailableMemoryForBackPropagation();

    // 0. get basic blocks with gradient operations
    // TODO: be more intelligent than duplicating the code for each block
    auto gradientBasicBlocks = getBasicBlocksWithGradientOperations(function);

    for(auto* gradientBasicBlock : gradientBasicBlocks)
    {
        // 1. partition the graph into wavefronts along the critical path
        auto wavefronts = partitionGraphIntoWavefronts(function, gradientBasicBlock);

        // 2. reserve the working set for forward prop and back prop
        double workingSpace = getForwardAndBackPropWorkingSpaceRequirement(wavefronts);

        availableMemory = std::max(0.0, availableMemory - workingSpace);

        // 3. divide the back propagation operation into phases
        auto phases = schedulePhases(wavefronts, availableMemory);

        auto insertionPoint = getInsertionPoint(gradientBasicBlock);

        // 4. generate operations for each phase
        generateOperations(function, phases, insertionPoint);
    }
}

StringSet MemoryEfficientBackPropagationPass::getRequiredAnalyses() const
{
    return StringSet({"OperationMemoryAnalysis"});
}

} // namespace optimization
} // namespace lucius




