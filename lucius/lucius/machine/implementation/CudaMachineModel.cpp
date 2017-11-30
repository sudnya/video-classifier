/*  \file   CudaMachineModel.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the CudaMachineModel class.
*/

// Lucius Include
#include <lucius/machine/interface/CudaMachineModel.h>

#include <lucius/parallel/interface/CudaDriver.h>

namespace lucius
{

namespace machine
{

double CudaMachineModel::getOperationLaunchOverhead() const
{
    return 20.0e-6;
}

double CudaMachineModel::getFloatingPointThroughput() const
{
    int clockRate = 0;
    int multiprocessorCount = 0;
    int majorVersion = 0;
    int minorVersion = 0;

    parallel::CudaDriver::cuDeviceGetAttribute(&clockRate,
        parallel::CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 0);
    parallel::CudaDriver::cuDeviceGetAttribute(&multiprocessorCount,
        parallel::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);

    parallel::CudaDriver::cuDeviceGetAttribute(&majorVersion,
        parallel::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
    parallel::CudaDriver::cuDeviceGetAttribute(&minorVersion,
        parallel::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);

    double ffmasPerMultiprocessor = 128.0;

    if((majorVersion == 6 && minorVersion == 0) ||
       (majorVersion == 7 && minorVersion == 0))
    {
        ffmasPerMultiprocessor = 64.0;
    }

    return clockRate * 1.0e3 * 2.0 * ffmasPerMultiprocessor * multiprocessorCount;
}

double CudaMachineModel::getMemoryOperationThroughput() const
{
    int memoryClockRate = 0;
    int memoryBusWidth = 0;

    parallel::CudaDriver::cuDeviceGetAttribute(&memoryClockRate,
        parallel::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, 0);
    parallel::CudaDriver::cuDeviceGetAttribute(&memoryBusWidth,
        parallel::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, 0);

    return memoryClockRate * 1.0e3 * memoryBusWidth / 8.0;
}

double CudaMachineModel::getNetworkOperationThroughput() const
{
    return 7.0e9;
}

double CudaMachineModel::getTotalSystemMemoryCapacity() const
{
    size_t bytes = 0;

    parallel::CudaDriver::cuDeviceTotalMem(&bytes, 0);

    return bytes;
}

}

}









