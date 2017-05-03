/*  \file   LazyIr.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the LazyIr interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/LazyIr.h>

#include <lucius/lazy-ir/interface/Context.h>

namespace lucius
{

namespace lazy
{

void newThreadLocalContext()
{
    getThreadLocalContext().clear();
}

thread_local Context context;

Context& getThreadLocalContext()
{
    return context;
}

IRBuilder& getBuilder()
{
    return getThreadLocalContext().getBuilder();
}

LazyValue getConstant(const Matrix& value)
{
    return getThreadLocalContext().getConstant(value);
}

LazyValue getConstant(int64_t integer)
{
    return getThreadLocalContext().getConstant(integer);
}

BasicBlock* newBasicBlock()
{
    return getThreadLocalContext().newBasicBlock();
}

void setBasicBlock(BasicBlock* block)
{
    getThreadLocalContext().setInsertionPoint(block);
}

}

}


