/*  \file   LazyValue.cpp
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The source file for the LazyValue class.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/LazyValue.h>

#include <lucius/runtime/interface/IRExecutionEngine.h>

namespace lucius
{

namespace lazy
{

LazyValue::LazyValue(ir::Value* value)
: _value(value)
{

}

ir::Value* LazyValue::getValue()
{
    return _value;
}

void* LazyValue::runProgram()
{
    auto program = getBuilder().getProgram();
    auto engine = IRExecutionEngineFactory::create(program.get());

    engine->run();

    return engine->getValueContents(getValue());
}

}

}



