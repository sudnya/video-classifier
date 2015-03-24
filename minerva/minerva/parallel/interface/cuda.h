
#pragma once

#ifdef __NVCC__
#define CUDA_DECORATOR __host__ __device__
#else
#define CUDA_DECORATOR
#endif


