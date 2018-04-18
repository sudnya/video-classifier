/*  \file   DynamicMemoryAllocationPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the DynamicMemoryAllocatin class.
*/

// Lucius Includes
#include <lucius/optimization/interface/DynamicMemoryAllocationPass.h>

#include <lucius/analysis/interface/PostDominatorAnalysis.h>
#include <lucius/analysis/interface/DominatorAnalysis.h>

#include <lucius/ir/target/interface/TargetOperation.h>
#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/AllocationOperation.h>
#include <lucius/ir/target/interface/FreeOperation.h>

#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/ops/interface/ControlOperation.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{
namespace optimization
{

// Namespace imports
using Operation = ir::Operation;
using ControlOperation = ir::ControlOperation;
using BasicBlock = ir::BasicBlock;
using TargetOperation = ir::TargetOperation;
using TargetValue = ir::TargetValue;
using AllocationOperation = ir::AllocationOperation;
using FreeOperation = ir::FreeOperation;
using PostDominatorAnalysis = analysis::PostDominatorAnalysis;
using DominatorAnalysis = analysis::DominatorAnalysis;

DynamicMemoryAllocationPass::DynamicMemoryAllocationPass()
: Pass("DynamicMemoryAllocationPass")
{
    // intentionally blank
}

DynamicMemoryAllocationPass::~DynamicMemoryAllocationPass()
{
    // intentionally blank
}

static bool isAllocatableValue(const TargetValue& value)
{
    return value.isTensor() || value.isInteger() || value.isFloat();
}

static bool needsAllocation(const TargetValue& value)
{
    return isAllocatableValue(value) && !value.isConstant();
}

static void insert(BasicBlock block, BasicBlock::iterator position, Operation newOperation)
{
    block.insert(position, newOperation);
}

static void addAllocation(TargetValue value, BasicBlock postDominator,
    BasicBlock::iterator insertionPoint)
{
    AllocationOperation allocation(value);

    util::log("DynamicMemoryAllocationPass") << "   adding allocation '"
        << allocation.toString() << "'.\n";

    insert(postDominator, insertionPoint, allocation);
}

static void addFree(TargetValue value, BasicBlock postDominator,
    BasicBlock::iterator insertionPoint)
{
    FreeOperation free(value);

    util::log("DynamicMemoryAllocationPass") << "   adding free '"
        << free.toString() << "'.\n";

    insert(postDominator, insertionPoint, free);
}

using BasicBlockVector = std::vector<BasicBlock>;

static BasicBlock getPostDominatorOfAllUses(TargetValue operand,
    PostDominatorAnalysis* postDominatorAnalysis)
{
    auto uses = operand.getUsesAndDefinitions();

    assert(!uses.empty());

    BasicBlock postDominator = uses.front().getBasicBlock();

    for(auto& use : uses)
    {
        BasicBlockVector useBlocks;

        useBlocks.push_back(use.getBasicBlock());

        auto operation = use.getOperation();

        // make sure values are freed after the branch is evaluated
        if(operation.isControlOperation())
        {
            auto controlOperation = ir::value_cast<ControlOperation>(operation);

            auto possibleTargets = controlOperation.getPossibleTargets();

            useBlocks.insert(useBlocks.end(), possibleTargets.begin(), possibleTargets.end());
        }

        for(auto& block : useBlocks)
        {
            postDominator = postDominatorAnalysis->getPostDominator(block, postDominator);
        }
    }

    return postDominator;
}

static BasicBlock getDominatorOfAllDefinitions(TargetValue operand,
    DominatorAnalysis* dominatorAnalysis)
{
    auto& definitions = operand.getDefinitions();

    assert(!definitions.empty());

    BasicBlock dominator = definitions.front().getBasicBlock();

    for(auto& definition : definitions)
    {
        dominator = dominatorAnalysis->getDominator(definition.getBasicBlock(), dominator);
    }

    return dominator;
}

static bool dominatorContainsDefinition(TargetValue value, BasicBlock dominator)
{
    auto& definitions = value.getDefinitions();

    for(auto& definition : definitions)
    {
        if(definition.getBasicBlock() == dominator)
        {
            return true;
        }
    }

    return false;
}

static BasicBlock::iterator getFirstDefinition(TargetValue value, BasicBlock dominator)
{
    auto possibleDefinition = dominator.begin();

    for(; possibleDefinition != dominator.end(); ++possibleDefinition)
    {
        auto possibleTargetDefinition = ir::value_cast<TargetOperation>(*possibleDefinition);

        if(possibleTargetDefinition.getOutputOperand().getValue() == value)
        {
            break;
        }
    }

    assert(possibleDefinition != dominator.end());

    return possibleDefinition;
}

static void addAllocations(TargetValue value, DominatorAnalysis* dominatorAnalysis)
{
    auto dominator = getDominatorOfAllDefinitions(value, dominatorAnalysis);

    BasicBlock::iterator insertionPoint;

    if(dominatorContainsDefinition(value, dominator))
    {
        insertionPoint = getFirstDefinition(value, dominator);
    }
    else
    {
        insertionPoint = dominator.end();
    }

    addAllocation(value, dominator, insertionPoint);
}

static bool postDominatorContainsUse(TargetValue value, BasicBlock postDominator)
{
    auto uses = value.getUsesAndDefinitions();

    for(auto& use : uses)
    {
        if(use.getBasicBlock() == postDominator)
        {
            return true;
        }
    }

    return false;
}

static BasicBlock::iterator getLastUse(TargetValue value, BasicBlock postDominator)
{
    auto possibleUse = postDominator.rbegin();

    for(; possibleUse != postDominator.rend(); ++possibleUse)
    {
        auto possibleTargetUse = ir::value_cast<TargetOperation>(*possibleUse);

        auto& operands = possibleTargetUse.getOperands();

        bool foundUse = false;

        for(auto& operand : operands)
        {
            if(operand.getValue() == value)
            {
                foundUse = true;
                break;
            }
        }

        if(foundUse)
        {
            break;
        }
    }

    assert(possibleUse != postDominator.rend());

    return (++possibleUse).base();
}

static bool isReturned(TargetValue value)
{
    util::log("DynamicMemoryAllocationPass") << "  checking if '"
        << value.toString() << "' is returned.\n";

    for(auto& use : value.getUses())
    {
        auto userValue = use.getParent().getValue();

        assert(userValue.isOperation());

        auto operation = ir::value_cast<Operation>(userValue);

        util::log("DynamicMemoryAllocationPass") << "   checking use by '"
            << operation.toString() << "'\n";

        if(operation.isReturn())
        {
            return true;
        }
    }

    return false;
}

static void addFrees(TargetValue value, PostDominatorAnalysis* postDominatorAnalysis)
{
    // don't free values that are returned, return will take care of it
    if(isReturned(value))
    {
        util::log("DynamicMemoryAllocationPass") << "  skipping frees for returned value '"
            << value.toString() << "'.\n";
        return;
    }

    util::log("DynamicMemoryAllocationPass") << "  adding free for value '"
        << value.toString() << "'.\n";

    auto postDominator = getPostDominatorOfAllUses(value, postDominatorAnalysis);

    BasicBlock::iterator insertionPoint;

    if(postDominatorContainsUse(value, postDominator))
    {
        insertionPoint = ++getLastUse(value, postDominator);
    }
    else
    {
        insertionPoint = postDominator.begin();
    }

    addFree(value, postDominator, insertionPoint);
}

using TargetValueVector = std::vector<TargetValue>;
using TargetValueSet = std::set<TargetValue>;

static TargetValueVector getTargetValues(ir::Function& function)
{
    TargetValueSet values;

    for(auto& block : function)
    {
        for(auto& operation : block)
        {
            auto targetOperation = ir::value_cast<TargetOperation>(operation);

            auto& operands = targetOperation.getOperands();

            for(auto& operand : operands)
            {
                values.insert(ir::value_cast<TargetValue>(operand.getValue()));
            }
        }
    }

    return TargetValueVector(values.begin(), values.end());
}

void DynamicMemoryAllocationPass::runOnFunction(ir::Function& function)
{
    auto* postDominatorAnalysis = dynamic_cast<PostDominatorAnalysis*>(
        getAnalysis("PostDominatorAnalysis"));
    assert(postDominatorAnalysis);

    auto* dominatorAnalysis = dynamic_cast<DominatorAnalysis*>(
        getAnalysis("DominatorAnalysis"));
    assert(dominatorAnalysis);

    util::log("DynamicMemoryAllocationPass") << "Running on function  '"
        << function.toString() << "'.\n";

    auto values = getTargetValues(function);

    for(auto& value : values)
    {
        if(needsAllocation(value))
        {
            util::log("DynamicMemoryAllocationPass") << " adding allocations and free for value '"
                << value.toString() << "'.\n";
            addAllocations(value, dominatorAnalysis);
            addFrees(value, postDominatorAnalysis);
        }
    }

    util::log("DynamicMemoryAllocationPass") << " function is now '"
        << function.toString() << "'.\n";
}

StringSet DynamicMemoryAllocationPass::getRequiredAnalyses() const
{
    return StringSet({"PostDominatorAnalysis", "DominatorAnalysis"});
}

} // namespace optimization
} // namespace lucius







