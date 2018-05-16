/*  \file   Utilities.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the IR Utilities set of functions.
*/

// Lucius Includes
#include <lucius/ir/interface/Utilities.h>

#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/InsertionPoint.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Program.h>
#include <lucius/ir/interface/BasicBlock.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace ir
{

InsertionPoint getFirstAvailableInsertionPoint(const Value& v)
{
    // TODO: Support values other than operations
    assert(v.isOperation());

    auto operation = value_cast<Operation>(v);

    // start after the value is defined
    auto iterator = ++operation.getIterator();
    auto block    = operation.getBasicBlock();

    // if this is a control operation that defines a value (i.e. a call), select the next block
    if(operation.isCall())
    {
        block = block.getNextBasicBlock();
        iterator = block.begin();
    }

    // advance past the last phi
    while(iterator != block.end() && iterator->isPHI())
    {
        ++iterator;
    }

    return InsertionPoint(block, iterator);
}

InsertionPoint getProgramExitPoint(const Program& p)
{
    auto engine = p.getEngineEntryPoint();

    auto block = engine.back();

    if(block.hasTerminator())
    {
        return InsertionPoint(block, --block.end());
    }

    return InsertionPoint(block, block.end());
}

InsertionPoint prepareBlockToAddTerminator(const InsertionPoint& point)
{
    auto  block    = point.getBasicBlock();
    auto& iterator = point.getIterator();

    assert(iterator == block.end() || !iterator->isPHI());

    // Just insert into an empty block
    if(block.empty())
    {
        return point;
    }

    // Just insert into not empty blocks with no terminator
    if((iterator == block.end()) && !block.back().isControlOperation())
    {
        return point;
    }

    // Split other blocks
    auto function = block.getFunction();

    auto fallthrough = function.insert(block.getNextBasicBlock().getIterator(), BasicBlock());

    moveOperations(InsertionPoint(fallthrough, fallthrough.end()), iterator, block.end());

    return InsertionPoint(block, block.end());
}

void moveOperations(const InsertionPoint& insertionPoint,
    const_operation_iterator begin, const_operation_iterator end)
{
    auto  block    = insertionPoint.getBasicBlock();
    auto& iterator = insertionPoint.getIterator();

    for(; begin != end; )
    {
        auto operation = *begin;

        ++begin;

        operation.detachFromBasicBlock();

        block.insert(iterator, operation);
    }
}

} // namespace ir
} // namespace lucius





