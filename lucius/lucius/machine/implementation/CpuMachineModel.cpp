/*  \file   CpuMachineModel.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the CpuMachineModel class.
*/

// Lucius Include
#include <lucius/machine/interface/CpuMachineModel.h>

#include <lucius/util/interface/SystemCompatibility.h>

namespace lucius
{

namespace machine
{

double CpuMachineModel::getOperationLaunchOverhead() const
{
    return 500.0e-9;
}

double CpuMachineModel::getFloatingPointThroughput() const
{
    return util::getFMAsPerClockPerCore() *
           2.0 *
           util::getMaxClockSpeed() *
           util::getHardwareThreadCount();
}

double CpuMachineModel::getMemoryOperationThroughput() const
{
    // TODO: calculate this
    return 20.0e9;
}

double CpuMachineModel::getNetworkOperationThroughput() const
{
    // TODO: calculate this
    return 7.0e9;
}

double CpuMachineModel::getTotalSystemMemoryCapacity() const
{
    // TODO: get total memory
    return util::getFreePhysicalMemory();
}

}

}










