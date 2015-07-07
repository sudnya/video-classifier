

#include <minerva/parallel/interface/cuda.h>
#include <minerva/parallel/interface/CudaRuntimeLibrary.h>

namespace minerva
{

namespace parallel
{

bool isCudaEnabled()
{
    return CudaRuntimeLibrary::loaded();
}

}
}

