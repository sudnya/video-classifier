/*  \file   TargetMachineFactory.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the TargetMachineFactory class.
*/

// Lucius Includes
#include <lucius/machine/interface/TargetMachineFactory.h>

#include <lucius/machine/interface/CudaTargetMachine.h>
#include <lucius/machine/interface/CpuTargetMachine.h>

#include <lucius/parallel/interface/CudaDriver.h>

namespace lucius
{

namespace machine
{

std::unique_ptr<TargetMachineInterface> TargetMachineFactory::create(ir::Context& context)
{
    if(parallel::CudaDriver::loaded())
    {
        return create(context, "CudaTargetMachine");
    }

    return create(context, "CpuTargetMachine");
}

std::unique_ptr<TargetMachineInterface> TargetMachineFactory::create(ir::Context& context,
    const std::string& machineName)
{
    if(machineName == "CudaTargetMachine")
    {
        return std::make_unique<CudaTargetMachine>(context);
    }
    else if (machineName == "CpuTargetMachine")
    {
        return std::make_unique<CpuTargetMachine>(context);
    }

    return std::unique_ptr<TargetMachineInterface>();
}

}

}









