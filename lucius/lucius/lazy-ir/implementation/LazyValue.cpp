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
#include <lucius/runtime/interface/IRExecutionEngineOptions.h>

#include <lucius/ir/interface/IRBuilder.h>
#include <lucius/ir/interface/InsertionPoint.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Variable.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Program.h>
#include <lucius/ir/interface/Module.h>

#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/Any.h>
#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace lazy
{

using Value = ir::Value;
using Variable = ir::Variable;
using ValueSet = std::set<Value>;

class LazyValueImplementation
{
public:
    LazyValueImplementation()
    {

    }

public:
    void addDefinition(ir::Value value)
    {
        assert(_definitions.empty() || !isConstant());

        // variables need explicit stores to represent definitions
        // normal values are tracked in the definitions set
        if(!_definitions.empty() && isVariable())
        {
            getBuilder().storeToVariable(Variable(getValue()), value);
        }
        else
        {
            _definitions.insert(value);
        }
    }

public:
    ir::Value getValueForRead()
    {
        if(isVariable())
        {
            return getBuilder().loadFromVariable(Variable(getValue()));
        }

        return getValue();
    }

    ir::Value getValue() const
    {
        assert(!_definitions.empty());

        return *_definitions.begin();
    }

    const ValueSet& getDefinitions() const
    {
        return _definitions;
    }

    bool isVariable() const
    {
        return getValue().isVariable();
    }

    bool isConstant() const
    {
        return getValue().isConstant();
    }

public:
    runtime::IRExecutionEngine& buildEngine(ir::Program& program)
    {
        _engine = runtime::IRExecutionEngineFactory::create(program);

        return *_engine;
    }

    void destroyEngine()
    {
        _engine.reset();
    }

private:
    ValueSet _definitions;

private:
    std::unique_ptr<runtime::IRExecutionEngine> _engine;

};

LazyValue::LazyValue()
: _implementation(std::make_shared<LazyValueImplementation>())
{
}

LazyValue::LazyValue(ir::Value value)
: LazyValue()
{
    registerLazyValue(*this);
    addDefinition(value);
}

LazyValue::LazyValue(ir::Variable variable)
: LazyValue()
{
    addDefinition(variable);
}

matrix::Matrix LazyValue::materialize()
{
    return materialize<matrix::Matrix>();
}

ir::Value LazyValue::getValueForRead()
{
    return _implementation->getValueForRead();
}

ir::Value LazyValue::getValue() const
{
    return _implementation->getValue();
}

bool LazyValue::isVariable() const
{
    return _implementation->isVariable();
}

void LazyValue::addDefinition(ir::Value newDefinition)
{
    _implementation->addDefinition(newDefinition);
}

const LazyValue::ValueSet& LazyValue::getDefinitions() const
{
    return _implementation->getDefinitions();
}

bool LazyValue::operator<(const LazyValue& right) const
{
    return getValue() < right.getValue();
}

void* LazyValue::_runProgram()
{
    auto returnValue = _implementation->getValueForRead();

    auto& program = getBuilder().getProgram();

    util::log("LazyValue") << " Cloning program.\n";

    ValueMap mappedValues;

    ir::Program augmentedProgram = program.cloneModuleAndTieVariables(mappedValues);

    augmentedProgram.getModule().setName("JustInTimeCompiledModule");

    if(util::isLogEnabled("LazyValue"))
    {
        util::log("LazyValue") << "  Cloned program:\n"
            << augmentedProgram.toString();
    }

    auto& engine = _implementation->buildEngine(augmentedProgram);

    engine.getOptions().addTargetIndependentOptimizationPass("ConvertLazyProgramToSSAPass",
        getLazyValues(mappedValues));
    engine.getOptions().addTargetIndependentOptimizationPass("LazyProgramCompleterPass");

    auto mappedReturnValueIterator = mappedValues.find(returnValue);
    assert(mappedReturnValueIterator != mappedValues.end());

    auto& mappedReturnValue = mappedReturnValueIterator->second;

    engine.saveValue(mappedReturnValue);

    engine.run();

    return engine.getValueContents(mappedReturnValue);
}

void LazyValue::_clearState()
{
    _implementation->destroyEngine();
}

}

}



