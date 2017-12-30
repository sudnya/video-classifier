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

#include <lucius/machine/cpu/interface/CpuTargetOperationFactory.h>

namespace lucius
{

namespace machine
{

CpuTargetMachine::~CpuTargetMachine()
{
    // intentionally blank
}

CpuTargetMachine::TableEntryVector CpuTargetMachine::getEntries() const
{
    TableEntryVector vector;

    vector.push_back(
        std::make_pair("call", TableEntry({TableOperationEntry("call", {TableOperandEntry(0)})}))
    );

    vector.push_back(
        std::make_pair("return", TableEntry({TableOperationEntry("return", {TableOperandEntry(0)})}))
    );

    return vector;
}

std::unique_ptr<ir::TargetOperationFactory> CpuTargetMachine::getOperationFactory() const
{
    return std::make_unique<cpu::CpuTargetOperationFactory>();
}

std::string CpuTargetMachine::name() const
{
    return "CpuTargetMachine";
}

}

}











