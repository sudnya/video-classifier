/*  \file   MachineModelFactory.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the MachineModelFactory class.
*/

// Lucius Includes
#include <lucius/machine/interface/MachineModelFactory.h>

#include <lucius/machine/interface/CpuMachineModel.h>
#include <lucius/machine/interface/CudaMachineModel.h>

#include <lucius/parallel/interface/CudaDriver.h>

namespace lucius
{

namespace machine
{

std::unique_ptr<MachineModelInterface> MachineModelFactory::create()
{
    if(parallel::CudaDriver::loaded())
    {
        return create("CudaMachineModel");
    }

    return create("CpuMachineModel");
}

std::unique_ptr<MachineModelInterface> MachineModelFactory::create(const std::string& machineName)
{
    if(machineName == "CudaMachineModel")
    {
        return std::make_unique<CudaMachineModel>();
    }
    else if(machineName == "CpuMachineModel")
    {
        return std::make_unique<CpuMachineModel>();
    }

    return std::unique_ptr<MachineModelInterface>();
}

}

}








