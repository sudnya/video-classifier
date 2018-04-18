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
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Program.h>
#include <lucius/ir/interface/InsertionPoint.h>

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

    }

    ir::Context& getContext()
    {
        return _context;
    }

public:
    void clear()
    {
        _builder.clear();
        invalidateAnalyses();
        _values.clear();
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

    analysis::Analysis& getAnalysis(const std::string& name)
    {
        return getAnalysisForFunction(getFunction(), name);
    }

    analysis::Analysis& getAnalysisForFunction(const Function& function, const std::string& name)
    {
        auto key = AnalysisKey(function, name);

        auto analysis = _analyses.find(key);

        if(analysis == _analyses.end())
        {
            analysis = _analyses.insert(std::make_pair(key, AnalysisFactory::create(name))).first;

            analysis->second->runOnFunction(function);
        }

        return *analysis->second;
    }

    void invalidateAnalyses()
    {
        _analyses.clear();
    }

    void convertProgramToSSA()
    {
        util::log("Context") << "Converting program to SSA:\n"
            << _builder.getProgram().toString() << "\n";

        _dropAllPHIs();

        _addPHIsForDefinedValues();

        _renameValuesToNearestDefinition();
    }

    void registerLazyValue(const LazyValue& value)
    {
        _values.push_back(value);
    }

private:
    void _dropAllPHIs()
    {
        auto& program = _builder.getProgram();

        for(auto& function : program)
        {
            for(auto& block : function)
            {
                block.erase(block.phiBegin(), block.phiEnd());
            }
        }
    }

    void _addPHIsForDefinedValues()
    {
        using LazyValueSet = std::set<LazyValue>;
        using FunctionLazyValueMap = std::map<Function, LazyValueSet>;

        FunctionLazyValueMap valuesInFunctions;

        for(auto& value : _values)
        {
            for(auto& definition : value.getDefinitions())
            {
                if(!definition.isOperation())
                {
                    continue;
                }

                auto operation = ir::value_cast<ir::Operation>(definition);

                valuesInFunctions[operation.getBasicBlock().getParent()].insert(value);
            }
        }

        for(auto& valuesInFunction : valuesInFunctions)
        {
            auto& function = valuesInFunction.first;
            auto& values   = valuesInFunction.second;

            auto& dominatorAnalysis = static_cast<DominatorAnalysis&>(
                getAnalysisForFunction(function, "DominatorAnalysis"));

            for(auto& value : values)
            {
                auto& definitions = value.getDefinitions();

                for(auto& definition : definitions)
                {
                    _addPHIsForDefinition(definition, definitions, value, dominatorAnalysis);
                }
            }
        }
    }

    void _addPHIsForDefinition(const ir::Value& value, const ValueSet& definitions,
        LazyValue lazyValue, const DominatorAnalysis& dominatorAnalysis)
    {
        auto operation = ir::value_cast<ir::Operation>(value);

        // insert phis for definition
        auto frontier = dominatorAnalysis.getDominanceFrontier(operation.getBasicBlock());

        util::log("Context") << "Checking phis in dominance frontier of "
            << operation.getBasicBlock().name() << " for definition '"
            << operation.toString() << "'\n";

        for(auto& block : frontier)
        {
            util::log("Context") << " checking dominance frontier block "
                << block.name() << "\n";
            // augment or insert a new phi
            auto phi = block.phiBegin();
            auto end = block.phiEnd();

            for(; phi != end; ++phi)
            {
                if(definitions.count(*phi) != 0)
                {
                    break;
                }
            }

            // skip an existing phi
            if(phi != end)
            {
                continue;
            }

            auto predecessors = block.getPredecessors();

            // add new phi
            ir::PHIOperation phiOperation;

            for(auto& predecessor : predecessors)
            {
                phiOperation.addIncomingValue(value, predecessor);
            }

            lazyValue.addDefinition(phiOperation);

            block.push_front(phiOperation);

            util::log("Context") << "  Adding phi " << block.front().toString()
                << " for definition " << value.toString() << "\n";

            if(block == operation.getBasicBlock())
            {
                continue;
            }

            // add PHIs for the iterated dominance frontier
            _addPHIsForDefinition(block.front(), definitions, lazyValue, dominatorAnalysis);
        }

    }

private:

    using ValueStack = std::stack<ir::Value>;
    using ValueDefinitionStack = std::map<ir::Value, ValueStack>;
    using ValueMap = std::map<ir::Value, ir::Value>;

    ValueMap _getValueMap()
    {
        ValueMap valueMap;
        util::log("Context") << " Creating map tracking definitions to the same values.\n";
        for(auto& value : _values)
        {
            auto& definitions = value.getDefinitions();

            if(definitions.size() == 1 && definitions.begin()->isConstant())
            {
                continue;
            }

            for(auto& definition : definitions)
            {
                valueMap[definition] = value.getValue();
                util::log("Context") << "  '" << definition.toString() << "' maps to '"
                    << value.getValue().toString() << "'\n";
            }
        }

        auto& program = _builder.getProgram();

        for(auto& function : program)
        {
            if(function.empty())
            {
                continue;
            }

            for(auto& block : function)
            {
                for(auto& operation : block)
                {
                    auto mappedValue = valueMap.find(operation);

                    if(mappedValue == valueMap.end())
                    {
                        valueMap.insert(std::make_pair(operation, operation));
                        util::log("Context") << "  '" << operation.toString() << "' maps to '"
                            << operation.toString() << "'\n";
                    }
                }
            }
        }

        return valueMap;

    }

    void _renameValuesToNearestDefinition()
    {
        util::log("Context") << "Renaming values to their dominating definitions.\n";

        auto valueMap = _getValueMap();

        auto& program = _builder.getProgram();

        for(auto& function : program)
        {
            if(function.empty())
            {
                continue;
            }

            auto& dominatorAnalysis = static_cast<DominatorAnalysis&>(
                getAnalysisForFunction(function, "DominatorAnalysis"));

            auto tree = dominatorAnalysis.getDominatorTree();

            ValueDefinitionStack definitionStack;

            util::log("Context") << "Renaming values for function '"
                << function.toString() << "'\n";

            _renameValuesInBlock(function.getEntryBlock(), tree, definitionStack, valueMap);
        }
    }

    /* Rename algorithm:
        - DFS through the dominator tree, keep a stack of definitions for each variable

            - Iterate through (non-phi) operations in the current block, pushing
              definitions onto the variable stack

            - Replace uses of values in each operation with the top of the stack

            - Iterate through successors of current block

                - Replace uses of values in phi operations with the top of the stack

            - Pop the stack for all definitions of values in this block
     */
    void _renameValuesInBlock(ir::BasicBlock block, const DominatorTree& dominatorTree,
        ValueDefinitionStack& stack, const ValueMap& valueMap)
    {
        // - Iterate through (non-phi) operations in the current block, pushing
        //   definitions onto the variable stack
        for(auto op = block.begin(); op != block.end(); ++op)
        {
            auto& operation = *op;

            if(!operation.isPHI())
            {
                for(size_t operandIndex = 0, operandCount = operation.size();
                    operandIndex < operandCount; ++operandIndex)
                {
                    auto& operand = operation.getOperand(operandIndex);

                    auto value = operand.getValue();

                    auto mappedValue = valueMap.find(value);

                    if(mappedValue == valueMap.end())
                    {
                        continue;
                    }

                    auto lastDefinition = stack.find(mappedValue->second);

                    // access to an undefined value
                    assert(lastDefinition != stack.end());

                    util::log("Context") << "   Renaming operand  '"
                        << operand.toString() << "' with '"
                        << lastDefinition->second.top().toString() << "'\n";

                    // - Replace uses of values in each operation with the top of the stack
                    operation.replaceOperand(operand, ir::Use(lastDefinition->second.top()));
                }
            }

            auto mappedValue = valueMap.find(operation);

            assert(mappedValue != valueMap.end());

            util::log("Context") << "  Pushing definition '"
                << operation.toString() << "' of value '"
                << mappedValue->second.toString() << "'\n";
            stack[mappedValue->second].push(operation);
        }

        auto successors = block.getSuccessors();

        // - Iterate through successors of current block
        for(auto& successor : successors)
        {
            for(auto operation = successor.phiBegin();
                operation != successor.phiEnd(); ++operation)
            {
                auto phi = ir::value_cast<ir::PHIOperation>(*operation);

                auto& predecessorBlocks = phi.getIncomingBasicBlocks();

                for(size_t i = 0; i < phi.size(); ++i)
                {
                    auto& predecessorBlock = predecessorBlocks[i];

                    if(predecessorBlock != block)
                    {
                        continue;
                    }

                    auto& operand = phi.getOperand(i);

                    auto value = operand.getValue();

                    auto mappedValue = valueMap.find(value);

                    if(mappedValue == valueMap.end())
                    {
                        continue;
                    }

                    auto lastDefinition = stack.find(mappedValue->second);

                    // access to an undefined value
                    if(lastDefinition == stack.end())
                    {
                        continue;
                    }

                    util::log("Context") << "   Renaming operand  '"
                        << operand.toString() << "' with '"
                        << lastDefinition->second.top().toString() << "'\n";

                    // - Replace uses of values in phi operations with the top of the stack
                    phi.replaceOperand(operand, ir::Use(lastDefinition->second.top()));
                }
            }
        }

        // recurse on dominator tree successors
        auto dominatedChildren = dominatorTree.find(block);

        assert(dominatedChildren != dominatorTree.end());

        for(auto& child : dominatedChildren->second)
        {
            _renameValuesInBlock(child, dominatorTree, stack, valueMap);
        }

        // - Pop the stack for all definitions of values in this block
        for(auto operation = block.phiEnd(); operation != block.end(); ++operation)
        {
            auto mappedValue = valueMap.find(*operation);

            assert(mappedValue != valueMap.end());

            util::log("Context") << "  Popping definition '"
                << stack[mappedValue->second].top().toString() << "' of value '"
                << mappedValue->second.toString() << "'\n";
            stack[mappedValue->second].pop();
        }
    }

private:
    ir::Context _context;

private:
    ir::IRBuilder _builder;

private:
    using AnalysisKey = std::pair<Function, std::string>;
    using AnalysisMap = std::map<AnalysisKey, std::unique_ptr<Analysis>>;

    AnalysisMap _analyses;

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

analysis::Analysis& Context::getAnalysis(const std::string& name)
{
    return _implementation->getAnalysis(name);
}

void Context::invalidateAnalyses()
{
    _implementation->invalidateAnalyses();
}

void Context::convertProgramToSSA()
{
    _implementation->convertProgramToSSA();
}

void Context::registerLazyValue(const LazyValue& value)
{
    _implementation->registerLazyValue(value);
}

}

}




