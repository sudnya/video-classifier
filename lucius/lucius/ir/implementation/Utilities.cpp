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

    // advance past the last phi
    while(iterator != operation.getBasicBlock().end() && iterator->isPHI())
    {
        ++iterator;
    }

    return InsertionPoint(operation.getBasicBlock(), iterator);
}

} // namespace ir
} // namespace lucius





