/*  \file   ConvertLazyProgramToSSAPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the ConvertLazyProgramToSSAPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/ConvertLazyProgramToSSAPass.h>

#include <lucius/optimization/interface/PassManager.h>

#include <lucius/analysis/interface/DominatorAnalysis.h>

#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Program.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/ops/interface/PHIOperation.h>

#include <lucius/util/interface/Any.h>
#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <map>
#include <set>
#include <stack>

namespace lucius
{
namespace optimization
{

ConvertLazyProgramToSSAPass::ConvertLazyProgramToSSAPass(const util::Any& any)
: ProgramPass("ConvertLazyProgramToSSAPass"), _mergedValues(any.get<MergedValueVector>())
{

}

ConvertLazyProgramToSSAPass::~ConvertLazyProgramToSSAPass()
{
    // intentionally blank
}

class MergedValue
{
public:
    MergedValue(const ir::Value& value)
    : _value(value)
    {
        addDefinition(value);
    }

public:
    void addDefinition(const ir::Value& value)
    {
        _definitions.insert(value);
    }

public:
    using ValueSet = std::set<ir::Value>;
    using       iterator = ValueSet::iterator;
    using const_iterator = ValueSet::const_iterator;

public:
    iterator begin()
    {
        return _definitions.begin();
    }

    const_iterator begin() const
    {
        return _definitions.begin();
    }

    iterator end()
    {
        return _definitions.end();
    }

    const_iterator end() const
    {
        return _definitions.end();
    }

    bool contains(const ir::Value& value)
    {
        return _definitions.count(value) > 0;
    }

    ir::Value& getValue()
    {
        return _value;
    }

    size_t size() const
    {
        return _definitions.size();
    }

private:
    ir::Value _value;
    ValueSet  _definitions;
};

using MergedValueVector = ConvertLazyProgramToSSAPass::MergedValueVector;
using DominatorAnalysis = analysis::DominatorAnalysis;
using DominatorTree     = DominatorAnalysis::DominatorTree;

class SSAConverter
{
public:
    SSAConverter(ir::Program& program, const MergedValueVector& mergedValues,
        PassManager& manager)
    : _program(program), _mergedValues(mergedValues), _manager(manager)
    {

    }

public:
    using ValueVector = std::vector<MergedValue>;
    using FunctionValueMap = std::map<Function, ValueVector>;

    void convertProgramToSSA()
    {
        util::log("ConvertLazyProgramToSSAPass") << "Converting program to SSA:\n"
            << _program.toString() << "\n";

        auto mappedValues = _addPHIsForDefinedValues();

        _renameValuesToNearestDefinition(mappedValues);
    }

private:
    FunctionValueMap _addPHIsForDefinedValues()
    {
        FunctionValueMap valuesInFunctions;

        util::log("ConvertLazyProgramToSSAPass") << "Creating merged values...\n";

        for(auto& valueSet : _mergedValues)
        {
            auto& value = valueSet.front();

            if(!value.isOperation())
            {
                continue;
            }

            if(value.isVariable())
            {
                continue;
            }

            MergedValue mergedValue(value);

            for(auto& definition : valueSet)
            {
                if(!definition.isOperation())
                {
                    continue;
                }

                mergedValue.addDefinition(definition);
            }

            auto operation = ir::value_cast<ir::Operation>(value);
            auto function  = operation.getBasicBlock().getParent();

            util::log("ConvertLazyProgramToSSAPass") << " adding merged value "
                << operation.toString() << " in function '" << function.name() << "{"
                << function.id() << "}'\n";

            valuesInFunctions[function].push_back(mergedValue);
        }

        for(auto& valuesInFunction : valuesInFunctions)
        {
            auto& function = valuesInFunction.first;
            auto& values   = valuesInFunction.second;

            auto& dominatorAnalysis = static_cast<const DominatorAnalysis&>(
                *_manager.getAnalysisForFunction(function, "DominatorAnalysis"));

            for(auto& mergedValue : values)
            {
                for(auto& definition : mergedValue)
                {
                    _addPHIsForDefinition(definition, mergedValue, dominatorAnalysis);
                }
            }
        }

        return valuesInFunctions;
    }

    void _addPHIsForDefinition(const ir::Value& value, MergedValue& mergedValue,
        const DominatorAnalysis& dominatorAnalysis)
    {
        auto operation = ir::value_cast<ir::Operation>(value);

        // insert phis for definition
        auto frontier = dominatorAnalysis.getDominanceFrontier(operation.getBasicBlock());

        util::log("ConvertLazyProgramToSSAPass") << "Checking phis in dominance frontier of "
            << operation.getBasicBlock().name() << " for definition '"
            << operation.toString() << "'\n";

        for(auto& block : frontier)
        {
            util::log("ConvertLazyProgramToSSAPass") << " checking dominance frontier block "
                << block.name() << "\n";
            // augment or insert a new phi
            auto phi = block.phiBegin();
            auto end = block.phiEnd();

            for(; phi != end; ++phi)
            {
                if(mergedValue.contains(*phi) != 0)
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
            ir::PHIOperation phiOperation(value.getType());

            for(auto& predecessor : predecessors)
            {
                phiOperation.addIncomingValue(value, predecessor);
            }

            block.push_front(phiOperation);

            mergedValue.addDefinition(phiOperation);

            util::log("ConvertLazyProgramToSSAPass") << "  Adding phi " << block.front().toString()
                << " for definition " << value.toString() << "\n";

            if(block == operation.getBasicBlock())
            {
                continue;
            }

            // add PHIs for the iterated dominance frontier
            _addPHIsForDefinition(block.front(), mergedValue, dominatorAnalysis);
        }
    }

private:
    using ValueStack = std::stack<ir::Value>;
    using ValueDefinitionStack = std::map<ir::Value, ValueStack>;
    using ValueMap = std::map<ir::Value, ir::Value>;

    ValueMap _getValueMap(FunctionValueMap& values)
    {
        ValueMap valueMap;
        util::log("ConvertLazyProgramToSSAPass")
            << " Creating map tracking definitions to the same values.\n";
        for(auto& functionValues : values)
        {
            for(auto& mergedValue : functionValues.second)
            {
                if(mergedValue.size() == 1 && mergedValue.getValue().isConstant())
                {
                    continue;
                }

                for(auto& definition : mergedValue)
                {
                    valueMap[definition] = mergedValue.getValue();
                    util::log("ConvertLazyProgramToSSAPass") << "  '" << definition.toString()
                        << "' maps to '" << mergedValue.getValue().toString() << "'\n";
                }
            }
        }

        for(auto& function : _program)
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
                        util::log("ConvertLazyProgramToSSAPass") << "  '" << operation.toString()
                            << "' maps to '" << operation.toString() << "'\n";
                    }
                }
            }
        }

        return valueMap;
    }

    void _renameValuesToNearestDefinition(FunctionValueMap& values)
    {
        util::log("ConvertLazyProgramToSSAPass")
            << "Renaming values to their dominating definitions.\n";

        auto valueMap = _getValueMap(values);

        for(auto& function : _program)
        {
            if(function.empty())
            {
                continue;
            }

            auto& dominatorAnalysis = static_cast<const DominatorAnalysis&>(
                *_manager.getAnalysisForFunction(function, "DominatorAnalysis"));

            auto tree = dominatorAnalysis.getDominatorTree();

            ValueDefinitionStack definitionStack;

            util::log("ConvertLazyProgramToSSAPass") << "Renaming values for function '"
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

                    util::log("ConvertLazyProgramToSSAPass") << "   Renaming operand  '"
                        << operand.toString() << "' with '"
                        << lastDefinition->second.top().toString() << "'\n";

                    // - Replace uses of values in each operation with the top of the stack
                    operation.replaceOperand(operand, ir::Use(lastDefinition->second.top()));
                }
            }

            auto mappedValue = valueMap.find(operation);

            assert(mappedValue != valueMap.end());

            util::log("ConvertLazyProgramToSSAPass") << "  Pushing definition '"
                << operation.toString() << "' of value '"
                << mappedValue->second.toString() << "'\n";
            stack[mappedValue->second].push(operation);
        }

        auto successors = block.getSuccessors();

        // - Iterate through successors of current block, replace uses of values in PHIs
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

                    util::log("ConvertLazyProgramToSSAPass") << "   Renaming operand  '"
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

            util::log("ConvertLazyProgramToSSAPass") << "  Popping definition '"
                << stack[mappedValue->second].top().toString() << "' of value '"
                << mappedValue->second.toString() << "'\n";
            stack[mappedValue->second].pop();
        }
    }

private:
    ir::Program& _program;

private:
    const MergedValueVector& _mergedValues;

private:
    PassManager& _manager;
};

void ConvertLazyProgramToSSAPass::runOnProgram(ir::Program& program)
{
    SSAConverter converter(program, _mergedValues, *getManager());

    converter.convertProgramToSSA();
}

StringSet ConvertLazyProgramToSSAPass::getRequiredAnalyses() const
{
    return {"DominatorAnalysis"};
}

} // namespace optimization
} // namespace lucius









