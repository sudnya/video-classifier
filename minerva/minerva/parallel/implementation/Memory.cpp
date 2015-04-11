

// Minerva Includes
#include <minerva/parallel/interface/Memory.h>

// Standard Library Includes
#include <cstdlib>

namespace minerva
{
namespace parallel
{

void* malloc(size_t size)
{
	#ifdef __NVCC__
	void* address = nullptr;
	CudaRuntime::cudaMallocManaged(&address, size);
	
	return address;
	#else
	return std::malloc(size);
	#endif
}

void free(void* address)
{
	#ifdef __NVCC__
	CudaRuntime::cudaFree(address);
	#else
	std::free(address);
	#endif
}

}
}




