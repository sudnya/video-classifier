/*  \file   MemoryEfficientBackPropagationPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the MemoryEfficientBackPropagationPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/MemoryEfficientBackPropagationPass.h>

#include <lucius/analysis/interface/OperationMemoryAnalysis.h>

#include <lucius/machine/interface/MachineModel.h>

#include <lucius/ir/ops/interface/ComputeGradientOperation.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/InsertionPoint.h>

#include <lucius/ir/ops/interface/OperationFactory.h>
#include <lucius/ir/ops/interface/BinaryApplyOperation.h>

#include <lucius/ir/values/interface/ConstantOperator.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <map>

namespace lucius
{
namespace optimization
{

MemoryEfficientBackPropagationPass::MemoryEfficientBackPropagationPass()
: FunctionPass("MemoryEfficientBackPropagationPass")
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

// Type definitions
using ComputeGradientOperation = ir::ComputeGradientOperation;
using Value = ir::Value;
using Operation = ir::Operation;
using BasicBlock = ir::BasicBlock;
using ValueSet = std::set<Value>;
using ValueSets = std::vector<ValueSet>;
using BasicBlockSet = std::set<BasicBlock>;
using OperationSet = std::set<Operation>;
using OperationMemoryAnalysis = analysis::OperationMemoryAnalysis;
using OperationFactory = ir::OperationFactory;

static ValueSet getGradientValues(const ir::BasicBlock& basicBlock)
{
    ValueSet gradientValues;

    for(auto& operation : basicBlock)
    {
        if(!operation.isGradientOperation())
        {
            continue;
        }

        gradientValues.insert(operation);
    }

    return gradientValues;
}

static ValueSet getGradientValues(ir::Function& function)
{
    ValueSet gradientValues;

    for(auto& block : function)
    {
        auto blockValues = getGradientValues(block);

        gradientValues.insert(blockValues.begin(), blockValues.end());
    }

    return gradientValues;
}

static BasicBlockSet getBasicBlocksWithGradientOperations(ir::Function& function)
{
    BasicBlockSet blocks;

    auto gradientValues = getGradientValues(function);

    for(auto& value : gradientValues)
    {
        auto operation = ir::value_cast<Operation>(value);

        blocks.insert(operation.getParent());
    }

    return blocks;
}

static OperationSet getOperationBackSlice(const ValueSet& gradientValues)
{
    OperationSet backSlice;
    OperationSet frontier;

    for(auto& value : gradientValues)
    {
        auto operation = ir::value_cast<Operation>(value);

        frontier.insert(operation);
    }

    while(!frontier.empty())
    {
        auto next = *frontier.begin();
        frontier.erase(frontier.begin());

        backSlice.insert(next);

        auto predecessors = next.getPredecessors();

        for(auto predecessor : predecessors)
        {
            // don't visit operations multiple times
            if(backSlice.count(predecessor) != 0)
            {
                continue;
            }

            // stop when we hit loads from variables
            if(predecessor.isLoad())
            {
                continue;
            }

            frontier.insert(predecessor);
        }
    }

    return backSlice;
}

static bool isReady(const Operation& operation, const OperationSet& finishedOperations)
{
    auto predecessors = operation.getPredecessors();

    for(auto predecessor : predecessors)
    {
        // skip loads
        if(predecessor.isLoad())
        {
            continue;
        }

        if(finishedOperations.count(predecessor) == 0)
        {
            return false;
        }
    }

    return true;
}

static OperationSet getReadyOperationsInBackSlice(const OperationSet& backSlice)
{
    OperationSet readyOperations;

    for(auto& operation : backSlice)
    {
        if(isReady(operation, {}))
        {
            util::log("MemoryEfficientBackPropagationPass") << "   added '"
                << operation.toString() << "'.\n";

            readyOperations.insert(operation);
        }
    }

    return readyOperations;
}

using OperationVector = std::vector<Operation>;
using OperationVectors = std::vector<OperationVector>;

static OperationVectors partitionGraphIntoWavefronts(ir::Function& function,
    const ir::BasicBlock& basicBlock)
{
    util::log("MemoryEfficientBackPropagationPass") << " Partitioning back slice into "
        "wave fronts.\n";

    OperationVectors wavefronts;

    auto gradientValues = getGradientValues(basicBlock);

    auto backSlice = getOperationBackSlice(gradientValues);

    auto readyOperations = getReadyOperationsInBackSlice(backSlice);

    OperationSet finishedOperations;

    // TODO: Consider making this a serial traversal to expose a longer chain for partitioning
    while(!readyOperations.empty())
    {
        util::log("MemoryEfficientBackPropagationPass") << "  new wave front "
            << wavefronts.size() << ".\n";

        OperationSet nextReadyOperations;

        wavefronts.push_back(OperationVector(readyOperations.begin(), readyOperations.end()));

        finishedOperations.insert(readyOperations.begin(), readyOperations.end());

        for(auto& operation : readyOperations)
        {
            auto successors = operation.getSuccessors();

            for(auto& successor : successors)
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

                if(nextReadyOperations.insert(successor).second)
                {
                    util::log("MemoryEfficientBackPropagationPass") << "   added '"
                        << successor.toString() << "'.\n";
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

        for(auto& operation : wavefront)
        {
            wavefrontWorkingSpaceSize += memoryAnalysis->getOperationMemoryRequirement(operation);
        }

        workingSpaceSize = std::max(workingSpaceSize, wavefrontWorkingSpaceSize);
    }

    return workingSpaceSize;
}

using IntVector = std::vector<size_t>;
using IntSet = std::set<size_t>;
using IntVectors = std::vector<IntVector>;

static double getSavedMemoryRequirement(const Operation& operation,
    const OperationMemoryAnalysis* memoryAnalysis)
{
    return memoryAnalysis->getOperationSavedMemoryRequirement(operation);
}

static double getMemoryUsage(const OperationVector& wavefront,
    const OperationMemoryAnalysis* memoryAnalysis)
{
    double usage = 0.0;

    for(auto& operation : wavefront)
    {
        usage += getSavedMemoryRequirement(operation, memoryAnalysis);
    }

    return 0.0;
}

using DoubleVector = std::vector<double>;

static DoubleVector getMemoryUsage(const OperationVectors& wavefronts,
    const OperationMemoryAnalysis* memoryAnalysis)
{
    DoubleVector usage;

    for(auto& wavefront : wavefronts)
    {
        usage.push_back(getMemoryUsage(wavefront, memoryAnalysis));
    }

    return usage;
}

static bool bisectSchedule(IntVector& currentSchedule, double& availableMemory,
    size_t startingWavefront, size_t remainingWavefrontCount,
    const DoubleVector& wavefrontMemoryUsage)
{
    // save the midpoint if there is enough memory
    size_t midpoint = startingWavefront + remainingWavefrontCount / 2;

    double requiredMemory = wavefrontMemoryUsage[midpoint];

    if(availableMemory < requiredMemory)
    {
        return false;
    }

    availableMemory -= requiredMemory;

    util::log("MemoryEfficientBackPropagationPass") << "   scheduled wavefront "
        << midpoint << ", remaining memory is " << availableMemory << "\n";

    currentSchedule.push_back(midpoint);

    // recurse right
    bool rightIsFullyScheduled = true;

    if(midpoint + 1 < remainingWavefrontCount)
    {
        size_t rightRemainingWavefrontCount = remainingWavefrontCount - midpoint - 1;

        rightIsFullyScheduled = bisectSchedule(currentSchedule, availableMemory, midpoint + 1,
            rightRemainingWavefrontCount, wavefrontMemoryUsage);
    }

    if(!rightIsFullyScheduled)
    {
        return false;
    }

    // recurse left if any exist
    if(startingWavefront == midpoint)
    {
        return true;
    }

    size_t leftRemainingWavefrontCount = midpoint - startingWavefront;

    bool leftIsFullyScheduled = bisectSchedule(currentSchedule, availableMemory,
        startingWavefront, leftRemainingWavefrontCount, wavefrontMemoryUsage);

    return leftIsFullyScheduled;
}

static void removeFinishedWavefronts(size_t& remainingWavefronts,
    const IntVector& currentSchedule)
{
    IntSet wavefrontsInCurrentSchedule(currentSchedule.begin(), currentSchedule.end());

    while(remainingWavefronts > 0)
    {
        if(wavefrontsInCurrentSchedule.count(remainingWavefronts) == 0)
        {
            break;
        }

        --remainingWavefronts;
    }
}

static ValueSets schedulePhases(const OperationVectors& wavefronts, double availableMemory,
    const OperationMemoryAnalysis* memoryAnalysis)
{
    // Values in the schedule represent saved wavefronts, e.g. a schedule of [1] for a wavefront
    // with [[add], [mul, div], [sin]] entries, represents that the [mul, div] values are saved
    IntVectors schedule;

    size_t remainingWavefronts = wavefronts.size();

    DoubleVector wavefrontMemoryUsage = getMemoryUsage(wavefronts, memoryAnalysis);

    util::log("MemoryEfficientBackPropagationPass") << " Creating back propagation phases.\n";

    while(remainingWavefronts > 0)
    {
        util::log("MemoryEfficientBackPropagationPass") << "  phase " << schedule.size() << "\n";

        double phaseAvailableMemory = availableMemory;

        IntVector currentSchedule;

        // schedule values in the last wavefront to ensure progress
        size_t lastWavefrontIndex = remainingWavefronts - 1;

        util::log("MemoryEfficientBackPropagationPass") << "   scheduled wavefront "
            << lastWavefrontIndex << ", remaining memory is " << availableMemory << "\n";

        currentSchedule.push_back(lastWavefrontIndex);

        phaseAvailableMemory -= std::min(phaseAvailableMemory,
            wavefrontMemoryUsage[lastWavefrontIndex]);

        --remainingWavefronts;

        // recursively greedy schedule
        bisectSchedule(currentSchedule, phaseAvailableMemory,
            0, remainingWavefronts, wavefrontMemoryUsage);

        schedule.push_back(currentSchedule);

        removeFinishedWavefronts(remainingWavefronts, currentSchedule);
    }

    // Return the set of values saved in each phase
    util::log("MemoryEfficientBackPropagationPass") << " Finding values needed in each phase.\n";
    ValueSets values;

    for(auto& phase : schedule)
    {
        util::log("MemoryEfficientBackPropagationPass") << "  For phase...\n";
        ValueSet valuesSavedThisPhase;

        for(auto& wavefrontId : phase)
        {
            util::log("MemoryEfficientBackPropagationPass") << "   From wavefront "
                << wavefrontId << "\n";
            auto& wavefront = wavefronts[wavefrontId];

            for(auto& operation : wavefront)
            {
                util::log("MemoryEfficientBackPropagationPass") << "    From operation "
                    << operation.toString() << "\n";

                auto values = operation.getUsedValues();

                for(auto& value : values)
                {
                    if(!value.isOperation())
                    {
                        continue;
                    }

                    util::log("MemoryEfficientBackPropagationPass") << "     value '"
                        << value.toString() << "' is needed.\n";
                    valuesSavedThisPhase.insert(value);
                }
            }
        }

        values.emplace_back(std::move(valuesSavedThisPhase));
    }

    return values;
}

using ValueMap = std::map<Value, Value>;

static bool alreadyExists(const Value& value, const ValueSet& availableValues,
    const ValueMap& savedValues, const ValueSet& frontier)
{
    if(availableValues.count(value) != 0)
    {
        return true;
    }

    if(savedValues.count(value) != 0)
    {
        return true;
    }

    if(frontier.count(value) != 0)
    {
        return true;
    }

    return false;
}

using ValueList = std::list<Value>;

static ValueList getPredecessors(const Value& value)
{
    if(value.isOperation())
    {
        auto operation = ir::value_cast<Operation>(value);

        auto predecessors = operation.getPredecessors();

        return ValueList(predecessors.begin(), predecessors.end());
    }

    return ValueList();
}

using InsertionPoint = ir::InsertionPoint;
using BasicBlockList = std::list<BasicBlock>;

static void insertSubgraph(InsertionPoint& insertionPoint, BasicBlockList& subgraph)
{
    auto function = insertionPoint.getFunction();

    function.insert(insertionPoint, subgraph);
}

using BasicBlockMap = std::map<BasicBlock, BasicBlock>;

static BasicBlock addOrGetBasicBlock(BasicBlockList& subgraph, BasicBlockMap& blockMap,
    const BasicBlock& block)
{
    auto existingBlock = blockMap.find(block);

    if(existingBlock == blockMap.end())
    {
        BasicBlock newBlock;

        auto successors = block.getSuccessors();

        for(auto& successor : successors)
        {
            auto successorMapping = blockMap.find(successor);

            if(successorMapping == blockMap.end())
            {
                continue;
            }
        }

        auto predecessors = block.getPredecessors();

        for(auto& predecessor : predecessors)
        {
            auto predecessorMapping = blockMap.find(predecessor);

            if(predecessorMapping == blockMap.end())
            {
                continue;
            }
        }

        existingBlock = blockMap.insert(std::make_pair(block, newBlock)).first;
    }

    return existingBlock->second;
}

static bool isReady(const Value& value, const ValueMap& savedValues,
    const ValueMap& recomputedValues)
{
    auto operation = ir::value_cast<Operation>(value);

    auto values = operation.getUsedValues();

    for(auto& predecessor : values)
    {
        if(predecessor.isConstant())
        {
            continue;
        }

        if(savedValues.count(predecessor) != 0)
        {
            continue;
        }

        if(recomputedValues.count(predecessor) != 0)
        {
            continue;
        }

        return false;
    }

    return true;
}

static Operation constructRecomputeOperation(const Operation& operation,
    const ValueMap& recomputedValues, const ValueMap& savedValues, const BasicBlock& parent)
{
    assertM(false, "Not implemented.");
}

static void addRecomputeOperations(ValueMap& recomputedValues,
    const ValueSet& valuesToRecompute, ValueMap& savedValues, InsertionPoint& insertionPoint)
{
    BasicBlockList subgraph;
    BasicBlockMap  blockMap;

    ValueSet readyValuesToRecompute;

    for(auto& value : valuesToRecompute)
    {
        if(isReady(value, savedValues, recomputedValues))
        {
            readyValuesToRecompute.insert(value);
        }
    }

    // Create the subgraph of operations that compute available values
    while(!readyValuesToRecompute.empty())
    {
        auto value = *readyValuesToRecompute.begin();
        readyValuesToRecompute.erase(readyValuesToRecompute.begin());

        util::log("MemoryEfficientBackPropagationPass") << "   recomputing '"
            << value.toString() << "'.\n";

        auto operation = ir::value_cast<Operation>(value);

        auto block = operation.getParent();

        auto newBlock = addOrGetBasicBlock(subgraph, blockMap, block);

        auto recomputeOperation = constructRecomputeOperation(operation,
            recomputedValues, savedValues, newBlock);

        recomputedValues[operation] = recomputeOperation;

        // add ready successors
        auto successors = operation.getSuccessors();

        for(auto& successor : successors)
        {
            if(isReady(successor, savedValues, recomputedValues))
            {
                readyValuesToRecompute.insert(value);
            }
        }
    }

    // Insert the new subgraph
    insertSubgraph(insertionPoint, subgraph);
}

static void saveValues(ValueMap& savedValues, const ValueMap& availableValues,
    const ValueSet& phase)
{
    ValueMap newSavedValues;

    for(auto& value : phase)
    {
        auto savedValue = savedValues.find(value);

        if(savedValue != savedValues.end())
        {
            newSavedValues[value] = savedValue->second;
            continue;
        }

        auto availableValue = availableValues.find(value);

        assert(availableValue != availableValues.end());

        newSavedValues[value] = availableValue->second;
    }

    savedValues = std::move(newSavedValues);
}

static bool canGradientBeComputed(const Value& connectedValue, const ValueMap& savedValues,
    const ValueMap& gradientValues)
{
    auto operation = ir::value_cast<Operation>(connectedValue);

    // it can be computed if all required operands are available
    auto descriptor = OperationFactory::getBackPropagationOperandDescriptor(operation);

    if(descriptor.needsSavedOutput())
    {
        if(savedValues.count(operation) == 0)
        {
            return false;
        }
    }

    for(auto& savedOperandIndex : descriptor)
    {
        auto value = operation.getOperand(savedOperandIndex).getValue();

        if(!value.isOperation())
        {
            continue;
        }

        if(savedValues.count(value) == 0)
        {
            return false;
        }
    }

    return true;
}

static void addConnectedValues(ValueSet& frontier, const Value& value,
    const ValueMap& gradientValues, const ValueMap& savedValues)
{
    auto connectedValues = getPredecessors(value);

    for(auto& connectedValue : connectedValues)
    {
        if(gradientValues.count(connectedValue) != 0)
        {
            continue;
        }

        if(!canGradientBeComputed(connectedValue, savedValues, gradientValues))
        {
            continue;
        }

        util::log("MemoryEfficientBackPropagationPass") << "     added '"
            << connectedValue.toString() << "'\n";

        frontier.insert(connectedValue);
    }
}

static void addBackPropagationBranch(BasicBlock& successor, BasicBlock& newBlock)
{
    assertM(false, "Not Implemented.");
}

static BasicBlock setupNewBasicBlockForBackPropagation(
    const BasicBlock& originalBasicBlock, BasicBlockList& subgraph, BasicBlockMap& basicBlockMap)
{
    auto newBlockIterator = subgraph.insert(subgraph.end(), BasicBlock());

    auto& newBlock = *newBlockIterator;

    basicBlockMap[originalBasicBlock] = newBlock;

    auto successors = originalBasicBlock.getSuccessors();

    // reverse edges, connect to successors
    for(auto& successor : successors)
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
    BasicBlockMap& basicBlockMap, const Operation op, const BasicBlock& originalBasicBlock)
{
    auto basicBlock = basicBlockMap.find(originalBasicBlock);

    if(basicBlock == basicBlockMap.end())
    {
        setupNewBasicBlockForBackPropagation(originalBasicBlock, subgraph, basicBlockMap);
    }

    basicBlock->second.push_back(std::move(op));
}

static ValueList setupBackPropagationOperands(const Operation& operation,
    const ValueMap& savedValues, const ValueMap& gradientValues)
{
    util::log("MemoryEfficientBackPropagationPass") << "       setting up operands\n";
    auto descriptor = OperationFactory::getBackPropagationOperandDescriptor(operation);

    ValueList operands;

    // 1) requested input operands
    for(auto operandIndex : descriptor)
    {
        auto& operand = operation.getOperand(operandIndex);

        if(operand.getValue().isOperation())
        {
            auto savedValue = savedValues.find(operand.getValue());

            assert(savedValue == savedValues.end());

            operands.push_back(savedValue->second);
        }
        else
        {
            operands.push_back(operand.getValue());
        }
    }

    // 2) output operand if necessary
    if(descriptor.needsSavedOutput())
    {
        auto savedOutputValue = savedValues.find(operation);

        assert(savedOutputValue == savedValues.end());

        operands.push_back(savedOutputValue->second);
    }

    // 3) output gradient
    auto gradientValue = gradientValues.find(operation);

    assert(gradientValue != gradientValues.end());

    operands.push_back(gradientValue->second);

    return operands;
}

static Value addGradientStepOperation(BasicBlockList& subgraph,
    BasicBlockMap& basicBlockMap, const Value& value, const ValueMap& savedValues,
    const ValueMap& gradientValues, const BasicBlock& originalBasicBlock)
{
    auto uses = value.getUses();

    Value gradientValue;

    util::log("MemoryEfficientBackPropagationPass") << "     adding gradient step "
        "operations for uses\n";

    for(auto& use : uses)
    {
        auto operation = use.getOperation();//ir::value_cast<Operation>(use.getValue());

        util::log("MemoryEfficientBackPropagationPass") << "      for use in '" <<
            operation.toString() << "'\n";

        auto operands = setupBackPropagationOperands(operation, savedValues, gradientValues);

        auto backPropOperation = OperationFactory::createBackPropagationOperation(operation);

        backPropOperation.setOperands(operands);

        attachOperationToSubgraph(subgraph, basicBlockMap, backPropOperation, originalBasicBlock);

        if(!gradientValue.isValid())
        {
            gradientValue = backPropOperation;
        }
        else
        {
            ir::BinaryApplyOperation addOperation(gradientValue, backPropOperation, ir::Add());

            attachOperationToSubgraph(subgraph, basicBlockMap, addOperation, originalBasicBlock);
        }
    }

    return gradientValue;
}

static void addBackpropagationOperations(ValueMap& gradientValues, const ValueMap& savedValues,
    InsertionPoint& insertionPoint)
{
    ValueSet finishedValues;
    ValueSet frontier;

    BasicBlockList subgraph;
    BasicBlockMap  basicBlockMap;

    util::log("MemoryEfficientBackPropagationPass") << "    creating frontier of values "
        "for which gradients can be computed\n";
    // Initialize frontier with values that can be computed immediately (that have no dependencies)
    for(auto& pair : gradientValues)
    {
        auto& gradientValue = pair.first;

        addConnectedValues(frontier, gradientValue, gradientValues, savedValues);
    }

    util::log("MemoryEfficientBackPropagationPass") << "    adding compute gradient operations\n";
    // add operations to compute gradients that are ready
    while(!frontier.empty())
    {
        auto value = *frontier.begin();
        frontier.erase(frontier.begin());

        auto operation = ir::value_cast<Operation>(value);

        auto basicBlock = operation.getParent();

        util::log("MemoryEfficientBackPropagationPass") << "     for '" << value.toString()
            << "'\n";

        auto gradient = addGradientStepOperation(subgraph, basicBlockMap, value,
            savedValues, gradientValues, basicBlock);

        util::log("MemoryEfficientBackPropagationPass") << "      gradient is '"
            << gradient.toString() << "'\n";

        gradientValues.emplace(value, gradient);

        addConnectedValues(frontier, value, gradientValues, savedValues);
    }

    // insert basic blocks
    insertSubgraph(insertionPoint, subgraph);
}

static void getBackSlice(ValueSet& valuesToRecompute,
    ValueMap& savedValues, const Value& value, Function& function)
{
    if(savedValues.count(value) != 0)
    {
        return;
    }

    util::log("MemoryEfficientBackPropagationPass") << "   adding values to recompute in back "
        "slice of '" << value.toString() << "'.\n";

    // add predecessors until saved values or available values are encountered
    ValueSet frontier;

    frontier.insert(value);

    while(!frontier.empty())
    {
        auto next = *frontier.begin();
        frontier.erase(frontier.begin());

        valuesToRecompute.insert(next);

        auto predecessors = getPredecessors(next);

        for(auto& predecessor : predecessors)
        {
            if(alreadyExists(predecessor, valuesToRecompute, savedValues, frontier))
            {
                continue;
            }

            util::log("MemoryEfficientBackPropagationPass") << "   added '"
                << predecessor.toString() << "'.\n";

            frontier.insert(predecessor);
        }
    }
}

static void generateOperations(ir::Function& function, const ValueSets& phases,
    InsertionPoint& insertionPoint)
{
    util::log("MemoryEfficientBackPropagationPass") << " initializing saved values.\n";
    ValueMap savedValues;

    // in the first phase, all necessary values are saved
    for(auto& value : phases.front())
    {
        util::log("MemoryEfficientBackPropagationPass") << "  adding saved value '"
            << value.toString() << "' -> '" << value.toString() << "'\n";
        savedValues.insert(std::make_pair(value, value));
    }

    util::log("MemoryEfficientBackPropagationPass") << " initializing gradient values.\n";
    ValueMap gradientValues;

    for(auto& operation : insertionPoint.getBasicBlock())
    {
        if(!operation.isGradientOperation())
        {
            continue;
        }

        auto gradientValue = operation.getOperand(1).getValue();

        util::log("MemoryEfficientBackPropagationPass") << "  added gradient value '"
            << gradientValue.toString() << "' -> '" << gradientValue.toString()
            << "' from operation '" << operation.toString() << "'\n";

        gradientValues.insert(std::make_pair(gradientValue, gradientValue));
    }

    util::log("MemoryEfficientBackPropagationPass") << " generating operations for "
        << phases.size() << " phases.\n";

    for(auto p = phases.begin(); p != phases.end(); ++p)
    {
        auto& phase = *p;

        util::log("MemoryEfficientBackPropagationPass") << "  new phase "
            << std::distance(phases.begin(), p) << "\n";

        ValueSet valuesToRecompute;

        for(auto value : phase)
        {
            getBackSlice(valuesToRecompute, savedValues, value, function);
        }

        util::log("MemoryEfficientBackPropagationPass") << "   adding recompute operations.\n";
        ValueMap recomputedValues;

        addRecomputeOperations(recomputedValues, valuesToRecompute, savedValues, insertionPoint);

        util::log("MemoryEfficientBackPropagationPass") << "   saving recomputed values.\n";
        saveValues(savedValues, recomputedValues, phase);

        util::log("MemoryEfficientBackPropagationPass") << "   adding back prop operations.\n";
        addBackpropagationOperations(gradientValues, savedValues, insertionPoint);
    }
}

static InsertionPoint getInsertionPoint(const BasicBlock& block)
{
    // Start before the first gradient operation
    for(auto& operation : block)
    {
        if(operation.isGradientOperation())
        {
            return InsertionPoint(operation);
        }
    }

    assertM(false, "Could not find any gradient operation.");

    return InsertionPoint();
}

void MemoryEfficientBackPropagationPass::runOnFunction(ir::Function& function)
{
    auto* memoryAnalysis = dynamic_cast<OperationMemoryAnalysis*>(
        getAnalysisForFunction(function, "OperationMemoryAnalysis"));

    // get memory available for back prop
    double availableMemory = getAvailableMemoryForBackPropagation();

    util::log("MemoryEfficientBackPropagationPass") << "For function: "
        << function.toString() << "\n";

    util::log("MemoryEfficientBackPropagationPass") << "Running on function  '"
        << function.name() << "' with " << (availableMemory/1.0e9) << " (GB) available.\n";

    // 0. get basic blocks with gradient operations
    // TODO: be more intelligent than duplicating the code for each block
    auto gradientBasicBlocks = getBasicBlocksWithGradientOperations(function);

    for(auto& gradientBasicBlock : gradientBasicBlocks)
    {
        util::log("MemoryEfficientBackPropagationPass") << " Running on basic block with "
            "'" << gradientBasicBlock.name() << "' gradient operations.\n";

        // 1. partition the graph into wavefronts along the critical path
        auto wavefronts = partitionGraphIntoWavefronts(function, gradientBasicBlock);

        // 2. reserve the working set for forward prop and back prop
        double workingSpace = getForwardAndBackPropWorkingSpaceRequirement(
            wavefronts, memoryAnalysis);

        availableMemory = std::max(0.0, availableMemory - workingSpace);

        util::log("MemoryEfficientBackPropagationPass") << " Working space requirement is "
            << (workingSpace/1.0e9) << " (GB), remaining memory for backprop is "
            << (availableMemory/1.0e9) << " (GB).\n";

        // 3. divide the back propagation operation into phases
        auto phases = schedulePhases(wavefronts, availableMemory, memoryAnalysis);

        auto insertionPoint = getInsertionPoint(gradientBasicBlock);

        // 4. generate operations for each phase
        generateOperations(function, phases, insertionPoint);
    }

    util::log("MemoryEfficientBackPropagationPass") << " function is now '"
        << function.toString() << "'.\n";
}

StringSet MemoryEfficientBackPropagationPass::getRequiredAnalyses() const
{
    return StringSet({"OperationMemoryAnalysis"});
}

} // namespace optimization
} // namespace lucius



