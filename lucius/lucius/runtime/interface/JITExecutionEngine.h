/*  \file   JITExecutionEngine.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the JITExecutionEngine class.
*/

#pragma once

// Lucius Includes
#include <lucius/runtime/interface/IRExecutionEngine.h>

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace runtime { class JITExecutionEngineImplementation; } }

namespace lucius
{

namespace runtime
{

/*! \brief An execution engine that evaluates a program in a straightforward way. */
class JITExecutionEngine : public IRExecutionEngine
{
public:
    JITExecutionEngine(Program& program);
    virtual ~JITExecutionEngine() final;

public:
    void run() final;

public:
    void* getValueContents(const Value& ) final;

public:
    void saveValue(const Value& v) final;

private:
    std::unique_ptr<JITExecutionEngineImplementation> _implementation;

};

} // namespace runtime
} // namespace lucius



