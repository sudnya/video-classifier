/*  \file   Loops.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the loop interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/Loops.h>
#include <lucius/lazy-ir/interface/LazyIr.h>
#include <lucius/lazy-ir/interface/LazyValue.h>
#include <lucius/lazy-ir/interface/CopyOperations.h>
#include <lucius/lazy-ir/interface/MatrixOperations.h>
#include <lucius/lazy-ir/interface/ComparisonOperations.h>
#include <lucius/lazy-ir/interface/Operators.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/IRBuilder.h>
#include <lucius/ir/interface/Value.h>

// Standard Library Includes
#include <functional>

namespace lucius
{

namespace lazy
{

// Namespace imports
using BasicBlock = ir::BasicBlock;
using IRBuilder  = ir::IRBuilder;

static void addConditionalBranch(IRBuilder& builder, LazyValue condition,
    const BasicBlock& fallthrough, const BasicBlock& target)
{
    builder.addConditionalBranch(condition.getValueForRead(), fallthrough, target);

    // TODO: be more precise
    //invalidateAnalyses();
}

void forLoop(std::function<LazyValue()> initializer,
    std::function<LazyValue(LazyValue)> condition,
    std::function<LazyValue(LazyValue)> increment,
    std::function<void()> addBody)
{
    auto& builder = getBuilder();

    BasicBlock preheader = builder.addBasicBlock();
    BasicBlock header    = builder.addBasicBlock();
    BasicBlock body      = builder.addBasicBlock();
    BasicBlock exit      = builder.addBasicBlock();

    // TODO: be more precise
    //invalidateAnalyses();

    builder.setInsertionPoint(preheader);
    LazyValue iterator = initializer();

    if(iterator.getValue().isConstant())
    {
        iterator = copy(iterator);
    }

    builder.setInsertionPoint(header);
    addConditionalBranch(builder, condition(iterator), body, exit);

    builder.setInsertionPoint(body);
    addBody();
    copy(iterator, increment(iterator));
    addConditionalBranch(builder, condition(iterator), body, exit);

    builder.setInsertionPoint(exit);
}

void forLoop(size_t iterations, std::function<void()> body)
{
    forLoop(
    []()
    {
        return LazyValue(getConstant(0));
    },
    [=](LazyValue iterator)
    {
        return lessThan(iterator, LazyValue(getConstant(iterations)));
    },
    [](LazyValue iterator)
    {
        return applyBinary(iterator, LazyValue(getConstant(1)), Add());
    },
    body);
}

}

}



