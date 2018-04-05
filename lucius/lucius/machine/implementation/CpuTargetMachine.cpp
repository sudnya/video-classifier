/*  \file   CpuTargetMachine.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the CpuTargetMachine class.
*/

// Lucius Includes
#include <lucius/machine/interface/CpuTargetMachine.h>

#include <lucius/machine/interface/TableEntry.h>
#include <lucius/machine/interface/TableOperationEntry.h>
#include <lucius/machine/interface/TableOperandEntry.h>
#include <lucius/machine/interface/VariableInputOperandsEntry.h>

#include <lucius/machine/cpu/interface/CpuTargetOperationFactory.h>

namespace lucius
{

namespace machine
{

CpuTargetMachine::CpuTargetMachine(Context& context)
: TargetMachineInterface(context)
{

}

CpuTargetMachine::~CpuTargetMachine()
{
    // intentionally blank
}

CpuTargetMachine::TableEntryVector CpuTargetMachine::getEntries() const
{
    TableEntryVector vector;

    vector.push_back(
        std::make_pair("call", TableEntry({TableOperationEntry("call",
            {TableOperandEntry(0, true), TableOperandEntry(0), VariableInputOperandsEntry(1)})}))
    );

    vector.push_back(
        std::make_pair("return", TableEntry({TableOperationEntry("return",
            {TableOperandEntry(0)})}))
    );

    vector.push_back(
        std::make_pair("return", TableEntry({TableOperationEntry("return", {})}))
    );

    vector.push_back(
        std::make_pair("binary-apply", TableEntry({TableOperationEntry("cpu-binary-apply",
            {TableOperandEntry(0, true), TableOperandEntry(0), TableOperandEntry(1),
             TableOperandEntry(2)})}))
    );

    return vector;
}

std::unique_ptr<ir::TargetOperationFactory> CpuTargetMachine::getOperationFactory() const
{
    return std::make_unique<cpu::CpuTargetOperationFactory>(_context);
}

std::string CpuTargetMachine::name() const
{
    return "CpuTargetMachine";
}

}

}











