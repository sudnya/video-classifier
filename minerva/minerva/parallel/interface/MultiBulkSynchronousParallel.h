
#pragma once

namespace minerva
{
namespace parallel
{
namespace detail
{

#ifdef __NVCC__

template<typename FunctionType>
__global__ void kernelLauncher(FunctionType function, ThreadGroup threadGroup)
{
	function(threadGroup):
}

template<typename FunctionType>
void launchCudaKernel(FunctionType function)
{
	size_t ctasPerSM = 0;
	size_t threads   = 128;

	checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(static_cast<int*>(&ctasPerSM), kernelLauncher<FunctionType>, threads, 0));

	size_t multiprocessorCount = 0;
	
	checkCudaErrors(cudaDeviceGetAttribute(static_cast<int*>(&multiprocessorCount), cudaDevAttrMultiProcessorCount, 0));

	size_t ctas = multiprocessorCount * ctasPerSM;

	kernelLauncher<<<ctas, threads>>>(f, ThreadGroup(ctas * threads));
}
#endif

}

template<typename FunctionType>
void multiBulkSynchronousParallel(FunctionType function)
{
	#ifdef __NVCC__
	detail::launchCudaKernel(function);
	#else
	function(ThreadGroup(1));
	#endif
}

}
}


