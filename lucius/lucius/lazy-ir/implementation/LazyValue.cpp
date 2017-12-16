/*  \file   LazyValue.cpp
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The source file for the LazyValue class.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/LazyValue.h>
#include <lucius/lazy-ir/interface/LazyIr.h>

#include <lucius/runtime/interface/IRExecutionEngine.h>
#include <lucius/runtime/interface/IRExecutionEngineFactory.h>

#include <lucius/ir/interface/IRBuilder.h>
#include <lucius/ir/interface/Program.h>

#include <lucius/matrix/interface/Matrix.h>

namespace lucius
{

namespace lazy
{

LazyValue::LazyValue(ir::Value value)
: _value(value)
{

}

matrix::Matrix LazyValue::materialize()
{
    return materialize<matrix::Matrix>();
}

ir::Value& LazyValue::getValue()
{
    return _value;
}

const ir::Value& LazyValue::getValue() const
{
    return _value;
}

void* LazyValue::_runProgram()
{
    auto& program = getBuilder().getProgram();
    auto engine = runtime::IRExecutionEngineFactory::create(program);

    engine->saveValue(getValue());

    engine->run();

    return engine->getValueContents(getValue());
}

}

}



