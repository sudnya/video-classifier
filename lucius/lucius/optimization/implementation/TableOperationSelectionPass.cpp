/*  \file   TableOperationSelectionPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the TableOperationSelectionPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/TableOperationSelectionPass.h>

namespace lucius
{
namespace optimization
{

TableOperationSelectionPass::TableOperationSelectionPass()
{
    // intetionally blank
}

TableOperationSelectionPass::~TableOperationSelectionPass()
{
    // intentionally blank
}

static void generateOperations(OperationList& targetOperations,
    ValueList& valueMap, Operation* operation)
{
    auto& entry = getTableEntryForOperation(operation);

    auto& operands = operation->getOperands();

    for(auto& targetOperationEntry : entry)
    {
        auto targetOperation = TargetOperationFactory::create(targetOperationEntry.name());

        for(auto& targetOperandEntry : targetOperationEntry.getOperandEntries())
        {
            if(targetOperandEntry.isExistingOperand())
            {
                auto* operand = operands[targetOperandEntry.getExistingOperandIndex()];

                auto mapping = valueMap.find(operand);

                assert(mapping != valueMap.end());

                targetOperation->push_back(mapping->second);
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
            valueMap[operation] = targetOperations.back();
        }
    }
}

void TableOperationSelectionPass::runOnFunction(ir::Function& function)
{
    ValueMap valueMap;
    OperationList oldOperations;

    for(auto& basicBlock : function)
    {
        OperationList newOperations;

        for(auto& operation : basicBlock)
        {
            generateOperations(newOperations, operation);
        }

        basicBlock->swapOperations(newOperations);

        oldOperations.splice(oldOperations.end(), newOperations);
    }
}

StringSet TableOperationSelectionPass::getRequiredAnalyses() const
{
    return StringSet();
}

} // namespace optimization
} // namespace lucius





