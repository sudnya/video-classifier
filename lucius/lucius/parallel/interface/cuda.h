
#pragma once

#ifdef __CUDA_ARCH__
#define CUDA_SHARED_DECORATOR __shared__
#else
#define CUDA_SHARED_DECORATOR static
#endif

#ifdef __NVCC__
#define CUDA_DECORATOR __host__ __device__
#define CUDA_MANAGED_DECORATOR __managed__
#define CUDA_GLOBAL_DECORATOR __global__
#else
#define CUDA_DECORATOR
#define CUDA_MANAGED_DECORATOR
#define CUDA_GLOBAL_DECORATOR
#endif

namespace lucius
{
namespace parallel
{

bool isCudaEnabled();

}
}


