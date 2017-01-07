
// Lucius Includes
#include <lucius/parallel/interface/Debug.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <iostream>

namespace lucius
{
namespace parallel
{

CUDA_DECORATOR bool isLogEnabled(const std::string& name)
{
    #ifdef __NVCC__
    return true;
    #else
    return util::isLogEnabled(name);
    #endif
}

#if LOGGING_ENABLED
CUDA_DECORATOR void print(const std::string& message)
{
    std::cout << message;
}

CUDA_DECORATOR LogStream::~LogStream()
{
    if(_isEnabled)
    {
        print(_message);
    }
}
#endif

}
}


