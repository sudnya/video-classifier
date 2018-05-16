/*  \file   OperationFinalizationPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the OperationFinalizationPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/OperationFinalizationPass.h>

#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/target/interface/TargetValue.h>

#include <lucius/machine/generic/interface/PHIOperation.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <map>
#include <stack>

namespace lucius
{
namespace optimization
{

class OperationFinalizationPassImplementation
{
public:
    void runOnFunction(const Function& function)
    {
        util::log("OperationFinalizationPass") << "  finalizing function '"
            << function.name() << "' for execution.\n";
        _function = function;

        _removePHIs(_function);

        util::log("OperationFinalizationPass") << "   function is: "
            << function.toString() << "\n";
    }

    Function getTargetFunction() const
    {
        return _function;
    }

private:
    void _removePHIs(Function& function)
    {
        std::map<ir::Value, std::set<ir::Value>> phiGraph;

        // create a graph of values connected through phis
        util::log("OperationFinalizationPass") << "  creating graph of connected phis.\n";
        for(auto& block : function)
        {
            for(auto& p : block.getPHIRange())
            {
                auto phi = ir::value_cast<machine::generic::PHIOperation>(p);

                std::vector<ir::Value> values;

                for(auto& operand : phi)
                {
                    values.push_back(operand.getValue());
                }

                for(auto& value : values)
                {
                    for(auto& connection : values)
                    {
                        if(value == connection)
                        {
                            continue;
                        }

                        phiGraph[value].insert(connection);
                        util::log("OperationFinalizationPass") << "  '" << value.toString()
                            << "' -> '" << connection.toString() << "'\n";
                    }
                }
            }
        }

        // Find all values reachable through the graph, assign them to the same value
        std::map<ir::Value, ir::Value> mappedValues;

        util::log("OperationFinalizationPass") << "  merging reachable nodes in the graph.\n";
        for(auto& node : phiGraph)
        {
            auto& rootValue = node.first;

            auto mappedValue = mappedValues.find(rootValue);

            if(mappedValue != mappedValues.end())
            {
                continue;
            }

            std::stack<ir::Value> valueStack;

            mappedValues[rootValue] = rootValue;
            valueStack.push(rootValue);

            util::log("OperationFinalizationPass") << "   creating value root at '"
                << rootValue.toString() << "'.\n";

            while(!valueStack.empty())
            {
                auto phiNode = phiGraph.find(valueStack.top());
                assert(phiNode != phiGraph.end());

                valueStack.pop();

                for(auto& connection : phiNode->second)
                {
                    auto connectionMapping = mappedValues.find(connection);

                    if(connectionMapping != mappedValues.end())
                    {
                        assert(connectionMapping->second == rootValue);
                        continue;
                    }

                    valueStack.push(connection);
                    mappedValues[connection] = rootValue;
                    util::log("OperationFinalizationPass") << "    mapping '"
                        << connection.toString() << "' -> '" << rootValue.toString() << "'.\n";
                }
            }
        }

        // Replace all uses
        for(auto& block : _function)
        {
            for(auto& operation : block.getNonPHIRange())
            {
                for(size_t o = 0; o < operation.size(); ++o)
                {
                    auto& operand = operation.getOperand(o);

                    auto mapping = mappedValues.find(operand.getValue());

                    if(mapping == mappedValues.end())
                    {
                        continue;
                    }

                    operation.replaceOperand(operand, ir::Use(mapping->second));
                }
            }
        }

        // Delete all phis
        for(auto& block : _function)
        {
            block.erase(block.phiBegin(), block.phiEnd());
        }
    }

private:
    Function _function;
};

OperationFinalizationPass::OperationFinalizationPass()
: FunctionPass("OperationFinalizationPass"),
  _implementation(std::make_unique<OperationFinalizationPassImplementation>())
{
    // intentionally blank
}

OperationFinalizationPass::~OperationFinalizationPass()
{
    // intentionally blank
}

void OperationFinalizationPass::runOnFunction(ir::Function& function)
{
    _implementation->runOnFunction(function);
}

StringSet OperationFinalizationPass::getRequiredAnalyses() const
{
    return StringSet();
}

Function OperationFinalizationPass::getTargetFunction()
{
    return _implementation->getTargetFunction();
}

} // namespace optimization
} // namespace lucius





