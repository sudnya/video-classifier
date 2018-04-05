/*  \file   CudaTargetMachine.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the CudaTargetMachine class.
*/

// Lucius Includes
#include <lucius/machine/interface/CudaTargetMachine.h>

#include <lucius/machine/interface/TableEntry.h>

#include <lucius/machine/cuda/interface/CudaTargetOperationFactory.h>

namespace lucius
{

namespace machine
{

CudaTargetMachine::CudaTargetMachine(Context& context)
: TargetMachineInterface(context)
{

}

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

std::unique_ptr<ir::TargetOperationFactory> CudaTargetMachine::getOperationFactory() const
{
    return std::make_unique<cuda::CudaTargetOperationFactory>(_context);
}

std::string CudaTargetMachine::name() const
{
    return "CudaTargetMachine";
}

}

}












