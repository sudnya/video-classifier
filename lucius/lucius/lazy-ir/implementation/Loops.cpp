/*  \file   Loops.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the loop interface functions.
*/

// Standard Library Includes
#include <function>

// Forward Declarations
class lucius { class lazy { class LazyValue; } }

namespace lucius
{

namespace lazy
{

void forLoop(std::function<LazyValue()> initializer,
    std::function<LazyValue(LazyValue)> condition,
    std::function<LazyValue(LazyValue)> increment,
    std::function<void()> addBody)
{
    auto& builder = getBuilder();

    BasicBlock* preheader = builder.newBasicBlock();
    BasicBlock* header    = builder.newBasicBlock();
    BasicBlock* body      = builder.newBasicBlock();
    BasicBlock* exit      = builder.newBasicBlock();

    builder.setInsertionPoint(preheader);
    LazyValue iterator = initializer();

    builder.setInsertionPoint(header);
    builder.addConditionalBranch(condition(iterator).getValue(), body, exit);

    builder.setInsertionPoint(body);
    addBody();
    copy(iterator, increment(iterator));
    builder.addConditionalBranch(condition(iterator).getValue(), body, exit);

    builder.setInsertionPoint(exit);
}

void forLoop(size_t iterations, std::function<void()> body)
{
    forLoop(
    []()
    {
        return getConstant(0);
    },
    [=](LazyValue iterator)
    {
        return lazy::lessThan(iterator, getConstant(iterations));
    },
    [](LazyValue iterator)
    {
        return lazy::apply(iterator, getConstant(1), Add());
    },
    body);
}

}

}



