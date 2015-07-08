

#include <lucious/parallel/interface/cuda.h>
#include <lucious/parallel/interface/CudaRuntimeLibrary.h>

namespace lucious
{

namespace parallel
{

bool isCudaEnabled()
{
    return CudaRuntimeLibrary::loaded();
}

}
}

