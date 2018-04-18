/*  \file   LazyIr.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the LazyIr interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/LazyIr.h>

#include <lucius/lazy-ir/interface/Context.h>
#include <lucius/lazy-ir/interface/LazyValue.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/IRBuilder.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace lazy
{

void newThreadLocalContext()
{
    getThreadLocalContext().clear();
}

void saveThreadLocalContext(std::ostream& stream)
{
    assertM(false, "Not implemented.");
}

void loadThreadLocalContext(std::istream& stream)
{
    assertM(false, "Not implemented.");
}

thread_local Context context;

Context& getThreadLocalContext()
{
    return context;
}

ir::IRBuilder& getBuilder()
{
    return getThreadLocalContext().getBuilder();
}

analysis::Analysis& getAnalysis(const std::string& name)
{
    return getThreadLocalContext().getAnalysis(name);
}

void invalidateAnalyses()
{
    return getThreadLocalContext().invalidateAnalyses();
}

void convertProgramToSSA()
{
    getThreadLocalContext().convertProgramToSSA();
}

void registerLazyValue(const LazyValue& value)
{
    getThreadLocalContext().registerLazyValue(value);
}

LazyValue getConstant(const matrix::Matrix& value)
{
    return getThreadLocalContext().getConstant(value);
}

LazyValue getConstant(int64_t integer)
{
    return getThreadLocalContext().getConstant(integer);
}

ir::BasicBlock newBasicBlock()
{
    return getThreadLocalContext().newBasicBlock();
}

void setBasicBlock(const ir::BasicBlock& block)
{
    getBuilder().setInsertionPoint(block);
}

size_t getHandle(LazyValue value)
{
    assertM(false, "Not implemented.");

    return 0;
}

LazyValue lookupValueByHandle(size_t handle)
{
    assertM(false, "Not implemented.");

    return LazyValue();
}

}

}


