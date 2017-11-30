/*  \file   CpuTargetMachine.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the CpuTargetMachine class.
*/

// Lucius Includes
#include <lucius/machine/interface/CpuTargetMachine.h>

#include <lucius/machine/interface/TableEntry.h>

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

    // TODO: add rules

    return vector;
}

}

}











