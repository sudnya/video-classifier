
#pragma once

#include <lucius/parallel/interface/ConcurrentCollectives.h>
#include <lucius/parallel/interface/Synchronization.h>

namespace lucius
{
namespace parallel
{
namespace detail
{

#ifdef __NVCC__

inline void checkCudaErrors(cudaError_t status)
{
    if(status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template<typename FunctionType>
__global__ void
__launch_bounds__(GroupLevelSize<2>::cudaSize(), 1)
kernelLauncher(FunctionType function)
{
    function(ThreadGroup(blockDim.x * gridDim.x, threadIdx.x + blockIdx.x * blockDim.x));
}

template<typename FunctionType>
void launchCudaKernel(FunctionType function)
{
    int ctasPerSM = 4;
    int threads   = GroupLevelSize<2>::cudaSize();

    int multiprocessorCount = 0;

    checkCudaErrors(cudaDeviceGetAttribute(&multiprocessorCount,
        cudaDevAttrMultiProcessorCount, 0));

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctasPerSM, kernelLauncher<FunctionType>, threads, 0));

    size_t ctas = std::min(static_cast<uint32_t>(multiprocessorCount * ctasPerSM),
        GroupLevelSize<2>::cudaMaxSize());

    kernelLauncher<<<ctas, threads>>>(function);
}
#endif

}

template<typename FunctionType>
void multiBulkSynchronousParallel(FunctionType function)
{
    if(isCudaEnabled())
    {
        #ifdef __NVCC__
        setNotSynchronized();

        detail::launchCudaKernel(function);
        #else
        function(ThreadGroup(1, 0));
        #endif
    }
    else
    {
        function(ThreadGroup(1, 0));
    }
}

}
}


