/*  \file   Program.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Program class.
*/

// Lucius Includes
#include <lucius/ir/interface/Program.h>

#include <lucius/ir/interface/Module.h>
#include <lucius/ir/interface/Function.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace ir
{

class ProgramImplementation
{
public:
    ProgramImplementation(Context& c)
    : _module(c)
    {

    }

public:
    Module& getModule()
    {
        return _module;
    }

    const Module& getModule() const
    {
        return _module;
    }

private:
    Module _module;
};

Program::Program(Context& c)
: _implementation(std::make_unique<ProgramImplementation>(c))
{

}

Program::Program(Program&& p)
: _implementation(std::move(p._implementation))
{

}

Program::~Program()
{
    // intentionally blank
}

Function Program::getInitializationEntryPoint() const
{
    return getModule().getFunction("InitializationEntryPoint");
}

Function Program::getDataProducerEntryPoint() const
{
    return getModule().getFunction("DataProducerEntryPoint");
}

Function Program::getCostFunctionEntryPoint() const
{
    return getModule().getFunction("CostFunctionEntryPoint");
}

Function Program::getForwardPropagationEntryPoint() const
{
    return getModule().getFunction("ForwardPropagationEntryPoint");
}

Function Program::getEngineEntryPoint() const
{
    return getModule().getFunction("EngineEntryPoint");
}

Function Program::getIsFinishedFunction() const
{
    return getModule().getFunction("IsFinishedFunction");
}

Module& Program::getModule()
{
    return _implementation->getModule();
}

const Module& Program::getModule() const
{
    return _implementation->getModule();
}

void Program::setForwardPropagationEntryPoint(Function&& f)
{
    f.setName("ForwardPropagationEntryPoint");

    getModule().addFunction(std::move(f));
}

void Program::clear()
{
    return getModule().clear();
}

Program Program::cloneModuleAndTieVariables()
{
    assertM(false, "Not implemented");

    return Program(getModule().getContext());
}

} // namespace ir
} // namespace lucius






