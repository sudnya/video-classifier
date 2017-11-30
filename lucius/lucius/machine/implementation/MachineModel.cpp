/*  \file   MachineModel.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the MachineModel class.
*/

// Lucius Includes
#include <lucius/machine/interface/MachineModel.h>

#include <lucius/machine/interface/MachineModelInterface.h>
#include <lucius/machine/interface/MachineModelFactory.h>

namespace lucius
{

namespace machine
{

std::unique_ptr<MachineModelInterface> _implementation;

static MachineModelInterface& getImplementation()
{
    if(!_implementation)
    {
        _implementation = MachineModelFactory::create();
    }

    return *_implementation;
}

double MachineModel::getOperationLaunchOverhead()
{
    return getImplementation().getOperationLaunchOverhead();
}

double MachineModel::getFloatingPointThroughput()
{
    return getImplementation().getFloatingPointThroughput();
}

double MachineModel::getMemoryOperationThroughput()
{
    return getImplementation().getMemoryOperationThroughput();
}

double MachineModel::getNetworkOperationThroughput()
{
    return getImplementation().getNetworkOperationThroughput();
}

double MachineModel::getTotalSystemMemoryCapacity()
{
    return getImplementation().getTotalSystemMemoryCapacity();
}

}

}







