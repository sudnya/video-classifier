
// Lucius Includes
#include <lucius/parallel/interface/Debug.h>

// Standard Library Includes
#include <iostream>

namespace lucius
{
namespace parallel
{

#if ENABLE_LOGGING

CUDA_MANAGED_DECORATOR LogDatabase* hostLogDatabase = nullptr;

#if defined(__NVCC__)
CUDA_GLOBAL_DECORATOR void allocateLogDatabase()
{
    logDatabase = new LogDatabase;
}

CUDA_GLOBAL_DECORATOR void enableSpecificLogDatabaseLog(const char* logName)
{
    logDatabase->enableSpecificLog(logName);
}
#endif

CUDA_DECORATOR LogDatabase* createAndGetLogDatabase()
{
    if(hostLogDatabase == nullptr)
    {
        hostLogDatabase = new LogDatabase;
    }

    return hostLogDatabase;
}

#endif

}
}


