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

#include <lucius/matrix/interface/Matrix.h>

namespace lucius
{

namespace lazy
{

using ValueSet = std::set<ir::Value>;

class LazyValueImplementation
{
public:
    LazyValueImplementation()
    {

    }

public:
    void addDefinition(ir::Value value)
    {
        assert(_definitions.empty() || !_definitions.begin()->isConstant());

        _definitions.insert(value);
    }

public:
    ir::Value getValueForRead()
    {
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
    convertProgramToSSA();

    auto& program = getBuilder().getProgram();

    auto& engine = _implementation->buildEngine(program);

    engine.getOptions().addTargetIndependentOptimizationPass("LazyProgramCompleterPass");

    engine.saveValue(getValueForRead());

    engine.run();

    return engine.getValueContents(getValueForRead());
}

void LazyValue::_clearState()
{
    _implementation->destroyEngine();
}

}

}



