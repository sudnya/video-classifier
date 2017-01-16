
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

}
}

#include <lucius/parallel/implementation/cstring.inl>


