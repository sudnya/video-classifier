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

// Standard Library Includes
#include <cassert>

namespace lucius
{
namespace optimization
{

// Namespace imports
using Operation = ir::Operation;
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

static bool needsAllocation(const TargetValue& value)
{
    return value.getType().isTensor();
}

static void insertBefore(BasicBlock block, BasicBlock::iterator position, Operation newOperation)
{
    block.insert(position, newOperation);
}

static void insertAfter(BasicBlock block, BasicBlock::iterator position, Operation newOperation)
{
    block.insert(++position, newOperation);
}

static void addAllocation(TargetValue value, BasicBlock postDominator,
    BasicBlock::iterator insertionPoint)
{
    AllocationOperation allocation(value);

    insertBefore(postDominator, insertionPoint, allocation);
}

static void addFree(TargetValue value, BasicBlock postDominator,
    BasicBlock::iterator insertionPoint)
{
    FreeOperation free(value);

    insertAfter(postDominator, insertionPoint, free);
}

static BasicBlock getPostDominatorOfAllUses(TargetValue operand,
    PostDominatorAnalysis* postDominatorAnalysis)
{
    auto& uses = operand.getUses();

    assert(!uses.empty());

    BasicBlock postDominator = uses.front().getParent();

    for(auto& use : uses)
    {
        postDominator = postDominatorAnalysis->getPostDominator(use.getParent(), postDominator);
    }

    return postDominator;
}

static BasicBlock getDominatorOfAllDefinitions(TargetValue operand,
    DominatorAnalysis* dominatorAnalysis)
{
    auto& definitions = operand.getDefinitions();

    assert(!definitions.empty());

    BasicBlock dominator = definitions.front().getParent();

    for(auto& definition : definitions)
    {
        dominator = dominatorAnalysis->getDominator(definition.getParent(), dominator);
    }

    return dominator;
}

static bool dominatorContainsDefinition(TargetValue value, BasicBlock dominator)
{
    auto& definitions = value.getDefinitions();

    for(auto& definition : definitions)
    {
        if(definition.getParent() == dominator)
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
    auto& uses = value.getUses();

    for(auto& use : uses)
    {
        if(use.getParent() == postDominator)
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

        auto& operands = possibleTargetUse.getAllOperands();

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

static void addFrees(TargetValue value, PostDominatorAnalysis* postDominatorAnalysis)
{
    auto postDominator = getPostDominatorOfAllUses(value, postDominatorAnalysis);

    BasicBlock::iterator insertionPoint;

    if(postDominatorContainsUse(value, postDominator))
    {
        insertionPoint = getLastUse(value, postDominator);
    }
    else
    {
        insertionPoint = postDominator.begin();
    }

    addFree(value, postDominator, insertionPoint);
}

using TargetValueVector = std::vector<TargetValue>;

static TargetValueVector getTargetValues(ir::Function& function)
{
    TargetValueVector values;

    for(auto& block : function)
    {
        for(auto& operation : block)
        {
            auto targetOperation = ir::value_cast<TargetOperation>(operation);

            auto& operands = targetOperation.getAllOperands();

            for(auto& operand : operands)
            {
                values.push_back(ir::value_cast<TargetValue>(operand.getValue()));
            }
        }
    }

    return values;
}

void DynamicMemoryAllocationPass::runOnFunction(ir::Function& function)
{
    auto* postDominatorAnalysis = dynamic_cast<PostDominatorAnalysis*>(
        getAnalysis("PostDominatorAnalysis"));
    auto* dominatorAnalysis = dynamic_cast<DominatorAnalysis*>(
        getAnalysis("DominatorAnalysis"));

    auto values = getTargetValues(function);

    for(auto& value : values)
    {
        if(needsAllocation(value))
        {
            addAllocations(value, dominatorAnalysis);
            addFrees(value, postDominatorAnalysis);
        }
    }
}

StringSet DynamicMemoryAllocationPass::getRequiredAnalyses() const
{
    return StringSet({"PostDominatorAnalysis"});
}

} // namespace optimization
} // namespace lucius







