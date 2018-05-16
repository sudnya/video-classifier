/*  \file   Context.cpp
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The source file for the Context class.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/Context.h>
#include <lucius/lazy-ir/interface/LazyValue.h>

#include <lucius/ir/interface/Context.h>
#include <lucius/ir/interface/IRBuilder.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Constant.h>
#include <lucius/ir/interface/Module.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Program.h>
#include <lucius/ir/interface/InsertionPoint.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/ops/interface/PHIOperation.h>

#include <lucius/analysis/interface/AnalysisFactory.h>
#include <lucius/analysis/interface/Analysis.h>

#include <lucius/analysis/interface/DominatorAnalysis.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <map>
#include <stack>

namespace lucius
{

namespace lazy
{

// Namespace Imports
using BasicBlock = ir::BasicBlock;
using IRBuilder = ir::IRBuilder;
using Matrix = matrix::Matrix;
using Function = ir::Function;
using AnalysisFactory = analysis::AnalysisFactory;
using Analysis = analysis::Analysis;
using ValueSet = LazyValue::ValueSet;
using DominatorAnalysis = analysis::DominatorAnalysis;
using DominatorTree = DominatorAnalysis::DominatorTree;

class ContextImplementation
{
public:
    ContextImplementation()
    : _builder(getContext())
    {
        clear();
    }

    ir::Context& getContext()
    {
        return _context;
    }

public:
    void clear()
    {
        _builder.clear();
        _values.clear();

        _builder.getProgram().getModule().setName("LazyProgram");
    }

public:
    BasicBlock newBasicBlock()
    {
        return _builder.addBasicBlock();
    }

    void setBasicBlock(const BasicBlock& block)
    {
        _builder.setInsertionPoint(block);
    }

public:
    LazyValue getConstant(const Matrix& value)
    {
        return LazyValue(_builder.addConstant(value));
    }

    LazyValue getConstant(int64_t integer)
    {
        return LazyValue(_builder.addConstant(integer));
    }

public:
    IRBuilder& getBuilder()
    {
        return _builder;
    }

    Function getFunction()
    {
        return getBuilder().getInsertionPoint()->getBasicBlock().getParent();
    }

    void registerLazyValue(const LazyValue& value)
    {
        _values.push_back(value);
    }

    Context::MergedValueVector getLazyValues(const Context::ValueMap& mappedValues)
    {
        Context::MergedValueVector mergedValues;

        for(auto& value : _values)
        {
            auto& definitions = value.getDefinitions();

            Context::MergedValue mergedValue;

            for(auto& definition : definitions)
            {
                auto mapping = mappedValues.find(definition);

                if(mapping == mappedValues.end())
                {
                    mergedValue.push_back(definition);
                }
                else
                {
                    mergedValue.push_back(mapping->second);
                }
            }

            mergedValues.push_back(mergedValue);
        }

        return mergedValues;
    }

private:
    ir::Context _context;

private:
    ir::IRBuilder _builder;

private:
    using LazyValueVector = std::vector<LazyValue>;

    LazyValueVector _values;

};

Context::Context()
: _implementation(std::make_unique<ContextImplementation>())
{

}

Context::~Context()
{
    // intentionally blank
}

ir::Context& Context::getContext()
{
    return _implementation->getContext();
}

void Context::clear()
{
    _implementation->clear();
}

LazyValue Context::getConstant(const Matrix& value)
{
    return _implementation->getConstant(value);
}

LazyValue Context::getConstant(int64_t integer)
{
    return _implementation->getConstant(integer);
}

BasicBlock Context::newBasicBlock()
{
    return _implementation->newBasicBlock();
}

void Context::setBasicBlock(const BasicBlock& block)
{
    _implementation->setBasicBlock(block);
}

IRBuilder& Context::getBuilder()
{
    return _implementation->getBuilder();
}

void Context::registerLazyValue(const LazyValue& value)
{
    _implementation->registerLazyValue(value);
}

Context::MergedValueVector Context::getLazyValues(const ValueMap& valueMap)
{
    return _implementation->getLazyValues(valueMap);
}

}

}




