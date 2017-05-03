/*  \file   IRExecutionEngine.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the IRExecutionEngine class.
*/

#pragma once

namespace lucius
{

namespace runtime
{

/*! \brief An interface for an engine that executes an IR program. */
class IRExecutionEngine
{
public:
    IRExecutionEngine(const Program& program);
    virtual ~IRExecutionEngine();

public:
    virtual void run() = 0;

public:
    void* getValueContents(const Value*);

protected:
    Program& getProgram();

private:
    Program& _program;

};

} // namespace runtime
} // namespace lucius


