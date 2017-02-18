
// Lucius Includes
#include <lucius/parallel/interface/Debug.h>
#include <lucius/parallel/interface/Synchronization.h>

// Standard Library Includes
#include <iostream>

namespace lucius
{
namespace parallel
{

#if ENABLE_LOGGING

CUDA_DEVICE_DECORATOR LogDatabase* deviceLogDatabase = nullptr;
LogDatabase* hostLogDatabase = nullptr;

CUDA_DECORATOR LogDatabase* createAndGetDeviceLogDatabase();

#if defined(__NVCC__)
CUDA_GLOBAL_DECORATOR void createDeviceLogDatabase()
{
    createAndGetDeviceLogDatabase();
}

CUDA_GLOBAL_DECORATOR void enableSpecificLogDatabaseLog(const char* logName)
{
    deviceLogDatabase->enableSpecificLog(logName);
}

CUDA_GLOBAL_DECORATOR void enableAllDeviceLogsKernel(bool shouldAllLogsBeEnabled)
{
    createAndGetDeviceLogDatabase()->enableAllLogs(shouldAllLogsBeEnabled);
}
#endif

void enableAllDeviceLogs(bool shouldAllLogsBeEnabled)
{
    #if defined(__NVCC__)
    createAndGetDeviceLogDatabase();
    setNotSynchronized();
    enableAllDeviceLogsKernel<<<1, 1>>>(shouldAllLogsBeEnabled);
    synchronize();
    #endif
}

void enableSpecificDeviceLog(const string& name)
{
    #ifdef __NVCC__
    createAndGetDeviceLogDatabase();

    char* data = reinterpret_cast<char*>(parallel::malloc(name.size() + 1));

    std::memcpy(data, name.c_str(), name.size() + 1);

    setNotSynchronized();
    enableSpecificLogDatabaseLog<<<1, 1>>>(data);

    parallel::free(data);
    synchronize();
    #endif
}

LogDatabase* createAndGetHostLogDatabase()
{
    if(hostLogDatabase == nullptr)
    {
        hostLogDatabase = new LogDatabase;
    }

    return hostLogDatabase;
}

CUDA_DECORATOR LogDatabase* createAndGetDeviceLogDatabase()
{
    #if defined(__CUDA_ARCH__)
    if(deviceLogDatabase == nullptr)
    {
        deviceLogDatabase = new LogDatabase;
    }

    return deviceLogDatabase;
    #elif defined(__NVCC__)
    setNotSynchronized();
    createDeviceLogDatabase<<<1, 1>>>();
    synchronize();

    return nullptr;
    #endif
}

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


