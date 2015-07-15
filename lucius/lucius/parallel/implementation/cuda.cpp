

#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/CudaRuntimeLibrary.h>

namespace lucius
{

namespace parallel
{

bool isCudaEnabled()
{
    return CudaRuntimeLibrary::loaded();
}

}
}

