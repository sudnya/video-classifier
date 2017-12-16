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

static void generateOperations(OperationList& targetOperations,
    ValueMap& valueMap, Operation& operation)
{
    auto& entry = TargetMachine::getTableEntryForOperation(operation);

    for(auto& targetOperationEntry : entry)
    {
        auto targetOperation = TargetOperationFactory::create(targetOperationEntry.name());

        for(auto& targetOperandEntry : targetOperationEntry)
        {
            if(targetOperandEntry.isExistingOperand())
            {
                auto& use = operation.getOperand(targetOperandEntry.getExistingOperandIndex());
                auto operand = use.getValue();

                auto mapping = valueMap.find(operand);

                assert(mapping != valueMap.end());

                targetOperation.appendOperand(mapping->second);
            }
            else
            {
                // TODO: handle other types of table entries
                assertM(false, "Not implemented.");
            }
        }

        targetOperations.push_back(std::move(targetOperation));

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
        OperationList newOperations;

        for(auto& operation : basicBlock)
        {
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





