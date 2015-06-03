
#pragma once

#include <minerva/parallel/interface/ConcurrentCollectives.h>

namespace minerva
{
namespace parallel
{
namespace detail
{

#ifdef __NVCC__

void checkCudaErrors(cudaError_t status)
{
    if(status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template<typename FunctionType>
__global__ void kernelLauncher(FunctionType function)
{
	function(ThreadGroup(blockDim.x * gridDim.x, threadIdx.x + blockIdx.x * blockDim.x));
}

template<typename FunctionType>
void launchCudaKernel(FunctionType function)
{
	int ctasPerSM = 0;
	int threads   = 128;

	checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&ctasPerSM, kernelLauncher<FunctionType>, threads, 0));

	int multiprocessorCount = 0;

	checkCudaErrors(cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, 0));

	size_t ctas = multiprocessorCount * ctasPerSM;

	kernelLauncher<<<ctas, threads>>>(function);
}
#endif

}

template<typename FunctionType>
void multiBulkSynchronousParallel(FunctionType function)
{
	#ifdef __NVCC__
	detail::launchCudaKernel(function);
	#else
	function(ThreadGroup(1, 0));
	#endif
}

}
}


