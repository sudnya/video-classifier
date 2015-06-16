

// Minerva Includes
#include <minerva/parallel/interface/Memory.h>
#include <minerva/parallel/interface/CudaRuntimeLibrary.h>

// Standard Library Includes
#include <cstdlib>

namespace minerva
{
namespace parallel
{

void* malloc(size_t size)
{
    if(CudaRuntimeLibrary::loaded())
    {
        #ifdef __APPLE__
        return CudaRuntimeLibrary::cudaHostAlloc(size);
        #else
        return CudaRuntimeLibrary::cudaMallocManaged(size);
        #endif
    }
    else
    {
	    return std::malloc(size);
    }
}

void free(void* address)
{
    if(CudaRuntimeLibrary::loaded())
    {
	    CudaRuntimeLibrary::cudaFree(address);
	}
    else
    {
        std::free(address);
    }
}

}
}




