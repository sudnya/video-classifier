
// Lucius Includes
#include <lucius/parallel/interface/Debug.h>

// Standard Library Includes
#include <iostream>

namespace lucius
{
namespace parallel
{

#if ENABLE_LOGGING

CUDA_DEVICE_DECORATOR LogDatabase* deviceLogDatabase = nullptr;
LogDatabase* hostLogDatabase = nullptr;

LogDatabase* createAndGetHostLogDatabase()
{
    if(hostLogDatabase == nullptr)
    {
        hostLogDatabase = new LogDatabase;
    }

    return hostLogDatabase;
}

#if defined(__NVCC__)
CUDA_GLOBAL_DECORATOR void createDeviceLogDatabase();
#endif

CUDA_DEVICE_DECORATOR LogDatabase* createAndGetDeviceLogDatabase()
{
    #if defined(__CUDA_ARCH__)
    if(deviceLogDatabase == nullptr)
    {
        deviceLogDatabase = new LogDatabase;
    }
    #elif defined(__NVCC__)
    createDeviceLogDatabase<<<1, 1>>>();
    #endif

    return deviceLogDatabase;
}

#if defined(__NVCC__)
CUDA_GLOBAL_DECORATOR void createDeviceLogDatabase()
{
    createAndGetDeviceLogDatabase();
}
#endif

#if defined(__NVCC__)
CUDA_GLOBAL_DECORATOR void enableSpecificLogDatabaseLog(const char* logName)
{
    deviceLogDatabase->enableSpecificLog(logName);
}
#endif

#if defined(__NVCC__)
CUDA_GLOBAL_DECORATOR void enableAllDeviceLogs(bool shouldAllLogsBeEnabled)
{
    createAndGetDeviceLogDatabase()->enableAllLogs(shouldAllLogsBeEnabled);
}
#endif

CUDA_DECORATOR LogDatabase* createAndGetLogDatabase()
{
    #if defined(__CUDA_ARCH__)
    return createAndGetDeviceLogDatabase();
    #else
    return createAndGetHostLogDatabase();
    #endif
}

#endif

}
}


