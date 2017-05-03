/*  \file   IRExecutionEngine.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the IRExecutionEngine class.
*/

// Lucius Includes
#include <lucius/runtime/interface/IRExecutionEngine.h>

namespace lucius
{

namespace runtime
{

IRExecutionEngine::IRExecutionEngine(Program& program)
: _program(program)
{

}

IRExecutionEngine::~IRExecutionEngine()
{
    // intentionally blank
}

void* IRExecutionEngine::getValueContents(const Value* value)
{
    return getProgram().getOutputValueContents(value);
}

Program& IRExecutionEngine::getProgram()
{
    return _program;
}

} // namespace runtime
} // namespace lucius



