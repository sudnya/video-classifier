/*  \file   TableOperationSelectionPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the TableOperationSelectionPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/TableOperationSelectionPass.h>

#include <lucius/machine/interface/TargetMachine.h>
#include <lucius/machine/interface/TableEntry.h>
#include <lucius/machine/interface/TableOperationEntry.h>
#include <lucius/machine/interface/TableOperandEntry.h>

#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Type.h>

#include <lucius/ir/values/interface/ValueFactory.h>

#include <lucius/ir/target/interface/TargetOperationFactory.h>
#include <lucius/ir/target/interface/TargetOperation.h>
#include <lucius/ir/target/interface/TargetValue.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <map>
#include <cassert>

namespace lucius
{
namespace optimization
{

TableOperationSelectionPass::TableOperationSelectionPass()
: Pass("TableOperationSelectionPass")
{
    // intetionally blank
}

TableOperationSelectionPass::~TableOperationSelectionPass()
{
    // intentionally blank
}

using Operation = ir::Operation;
using OperationList = std::list<Operation>;
using Value = ir::Value;

using TargetValue = ir::TargetValue;
using TargetMachine = machine::TargetMachine;
using TargetOperationFactory = ir::TargetOperationFactory;

using ValueMap = std::map<Value, TargetValue>;


static bool isNormalValue(const Value& v)
{
    return v.isOperation();
}

static bool isNotTranslatedValue(const Value& v)
{
    return v.isFunction() || v.isConstant();
}

static void generateOperations(OperationList& targetOperations,
    ValueMap& valueMap, Operation& operation)
{
    auto& entry = TargetMachine::getTableEntryForOperation(operation);
    auto& operationFactory = TargetMachine::getFactory();

    for(auto& targetOperationEntry : entry)
    {
        auto targetOperation = operationFactory.create(targetOperationEntry.name());

        assert(targetOperation.isValid());

        for(auto& targetOperandEntry : targetOperationEntry)
        {
            if(targetOperandEntry.isOutput())
            {
                auto value = ir::ValueFactory::createValueForType(operation.getType());

                targetOperation.setOutputOperand(TargetValue(value));
            }
            else if(targetOperandEntry.isExistingOperand())
            {
                auto& use = operation.getOperand(targetOperandEntry.getExistingOperandIndex());
                auto operand = use.getValue();

                if(isNormalValue(operand))
                {
                    auto mapping = valueMap.find(operand);

                    assert(mapping != valueMap.end());

                    targetOperation.appendOperand(mapping->second);
                }
                else if(isNotTranslatedValue(operand))
                {
                    targetOperation.appendOperand(TargetValue(operand));
                }
                else
                {
                    // TODO: handle other types of operands
                    assertM(false, "Not implemented.");
                }
            }
            else
            {
                // TODO: handle other types of table entries
                assertM(false, "Not implemented.");
            }
        }

        util::log("TableOperationSelectionPass") << "   added new operation "
            << targetOperation.toString() << "\n";

        targetOperations.push_back(targetOperation);

        if(targetOperationEntry.isOutput())
        {
            valueMap[operation] = ir::value_cast<TargetValue>(
                targetOperation.getOutputOperand().getValue());
        }
    }
}

void TableOperationSelectionPass::runOnFunction(ir::Function& function)
{
    util::log("TableOperationSelectionPass") << "Running on function " << function.name() << "\n";

    ValueMap newValues;

    for(auto& basicBlock : function)
    {
        util::log("TableOperationSelectionPass") << " Running on basic block "
            << basicBlock.toSummaryString() << "\n";
        OperationList newOperations;

        for(auto& operation : basicBlock)
        {
            util::log("TableOperationSelectionPass") << "  Running on operation "
                << operation.toString() << "\n";
            generateOperations(newOperations, newValues, operation);
        }

        basicBlock.setOperations(std::move(newOperations));
    }

    util::log("TableOperationSelectionPass") << " new function is " << function.toString() << "\n";
}

StringSet TableOperationSelectionPass::getRequiredAnalyses() const
{
    return StringSet();
}

} // namespace optimization
} // namespace lucius





