
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>

// Standard Library Includes
#include <cstddef>

namespace lucius
{
namespace parallel
{

CUDA_DECORATOR inline size_t strlen(const char*);

CUDA_DECORATOR inline void* memcpy(void* dest, const void* src, size_t count);

CUDA_DECORATOR inline char* itoa(char* output_buff, unsigned long long int num);
CUDA_DECORATOR inline char* itoa(char* output_buff, long long int num);
CUDA_DECORATOR inline char* dtoa(char* output_buff, double num, size_t maxIntegerDigits,
    size_t maxDecimalDigits);
CUDA_DECORATOR inline char* itoh(char* output_buff, long long int num);

}
}

#include <lucius/parallel/implementation/cstring.inl>


