/*  \file   CudaTargetMachine.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the CudaTargetMachine class.
*/

// Lucius Includes
#include <lucius/machine/interface/CudaTargetMachine.h>

#include <lucius/machine/interface/TableEntry.h>

namespace lucius
{

namespace machine
{

CudaTargetMachine::~CudaTargetMachine()
{
    // intentionally blank
}

CudaTargetMachine::TableEntryVector CudaTargetMachine::getEntries() const
{
    TableEntryVector vector;

    // TODO: add rules

    return vector;
}

}

}












