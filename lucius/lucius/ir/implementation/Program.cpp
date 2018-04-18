/*  \file   Program.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Program class.
*/

// Lucius Includes
#include <lucius/ir/interface/Program.h>

#include <lucius/ir/interface/Module.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Variable.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <map>

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
: _implementation(std::make_unique<ProgramImplementation>(p.getContext()))
{
    std::swap(_implementation, p._implementation);
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

Function Program::getForwardPropagationEntryPoint() const
{
    return getModule().getFunction("ForwardPropagationEntryPoint");
}

Function Program::getEngineEntryPoint() const
{
    return getModule().getFunction("EngineEntryPoint");
}

Function Program::getIsFinishedEntryPoint() const
{
    return getModule().getFunction("IsFinishedEntryPoint");
}

Module& Program::getModule()
{
    return _implementation->getModule();
}

const Module& Program::getModule() const
{
    return _implementation->getModule();
}

void Program::setInitializationEntryPoint(Function f)
{
    f.setName("InitializationEntryPoint");

    getModule().addFunction(f);
}

void Program::setDataProducerEntryPoint(Function f)
{
    f.setName("DataProducerEntryPoint");

    getModule().addFunction(f);
}

void Program::setForwardPropagationEntryPoint(Function f)
{
    f.setName("ForwardPropagationEntryPoint");

    getModule().addFunction(f);
}

void Program::setEngineEntryPoint(Function f)
{
    f.setName("EngineEntryPoint");

    getModule().addFunction(f);
}

void Program::setIsFinishedEntryPoint(Function f)
{
    f.setName("IsFinishedEntryPoint");

    getModule().addFunction(f);
}

Program::iterator Program::begin()
{
    return getModule().begin();
}

Program::const_iterator Program::begin() const
{
    return getModule().begin();
}

Program::iterator Program::end()
{
    return getModule().end();
}

Program::const_iterator Program::end() const
{
    return getModule().end();
}

void Program::clear()
{
    return getModule().clear();
}

Program Program::cloneModuleAndTieVariables()
{
    auto program = Program(getModule().getContext());

    std::map<Value, Value> functionMap;

    // add functions
    for(auto& function : getModule())
    {
        auto newFunction = function.clone();

        functionMap[function] = newFunction;

        program.getModule().addFunction(newFunction);
    }

    // add variables, but don't clone them
    for(auto& variable : getModule().getVariables())
    {
        program.getModule().addVariable(variable);
    }

    // replace uses of cloned functions
    for(auto& function : program.getModule())
    {
        for(auto& block : function)
        {
            for(auto& operation : block)
            {
                for(auto operandPosition = operation.begin();
                         operandPosition != operation.end(); )
                {
                    auto& operand = *operandPosition;

                    ++operandPosition;

                    if(operand.getValue().isFunction())
                    {
                        auto clonedFunction = functionMap.find(operand.getValue());

                        if(clonedFunction != functionMap.end())
                        {
                            operation.replaceOperand(operand, Use(clonedFunction->second));
                        }
                    }
                }
            }
        }
    }

    return program;
}

Context& Program::getContext()
{
    return getModule().getContext();
}

std::string Program::toString() const
{
    return getModule().toString();
}

} // namespace ir
} // namespace lucius






