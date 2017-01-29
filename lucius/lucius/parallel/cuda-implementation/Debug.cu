
// Lucius Includes
#include <lucius/parallel/interface/Debug.h>

// Standard Library Includes
#include <iostream>

namespace lucius
{
namespace parallel
{

#if ENABLE_LOGGING

#if defined(__NVCC__)
CUDA_GLOBAL_DECORATOR void allocateLogDatabase()
{
    logDatabase = new LogDatabase;
}

CUDA_GLOBAL_DECORATOR void enableSpecificLogDatabaseLog(const char* logName)
{
    logDatabase->enableSpecificLog(logName);
}
#else
LogDatabase* logDatabase = nullptr;
#endif

#endif

}
}


