/*  \file   IRExecutionEngineFactory.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the IRExecutionEngineFactory class.
*/

// Lucius Includes
#include <lucius/runtime/interface/IRExecutionEngineFactory.h>

#include <lucius/runtime/interface/JITExecutionEngine.h>

namespace lucius
{
namespace runtime
{

std::unique_ptr<IRExecutionEngine> IRExecutionEngineFactory::create(ir::Program& program)
{
    return std::make_unique<JITExecutionEngine>(program);
}

} // namespace optimization
} // namespace lucius








