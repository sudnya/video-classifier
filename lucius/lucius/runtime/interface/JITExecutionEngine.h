/*  \file   JITExecutionEngine.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the JITExecutionEngine class.
*/

#pragma once

// Lucius Includes
#include <lucius/runtime/interface/IRExecutionEngine.h>

namespace lucius
{

namespace runtime
{

/*! \brief An execution engine that evaluates a program in a straightforward way. */
class JITExecutionEngine : public IRExecutionEngine
{
public:
    JITExecutionEngine(const Program& program);
    virtual ~JITExecutionEngine();

public:
    final void run();

};

} // namespace runtime
} // namespace lucius



