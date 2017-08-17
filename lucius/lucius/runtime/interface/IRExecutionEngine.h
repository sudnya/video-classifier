/*  \file   IRExecutionEngine.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the IRExecutionEngine class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class Program; } }
namespace lucius { namespace ir { class Value;   } }

namespace lucius
{

namespace runtime
{

/*! \brief An interface for an engine that executes an IR program. */
class IRExecutionEngine
{
public:
    using Value   = ir::Value;
    using Program = ir::Program;

public:
    IRExecutionEngine(Program& program);
    virtual ~IRExecutionEngine();

public:
    virtual void run() = 0;

public:
    virtual void* getValueContents(const Value& v) = 0;

protected:
    Program& getProgram();

private:
    Program& _program;

};

} // namespace runtime
} // namespace lucius


