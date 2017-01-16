
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cstring.h>

// Standard Library Includes
#include <cstdint>

namespace lucius
{
namespace parallel
{

CUDA_DECORATOR inline size_t strlen(const char* string)
{
    const char* position = string;

    while(*position != '\0')
    {
        ++position;
    }

    return position - string;
}

CUDA_DECORATOR inline void* memcpy(void* dest, const void* src, size_t count)
{
    //TODO: more efficient

    const int8_t* source      = reinterpret_cast<const int8_t*>(src);
          int8_t* destination = reinterpret_cast<      int8_t*>(dest);

    for(size_t i = 0; i < count; ++i)
    {
        destination[i] = source[i];
    }

    return dest;
}

}
}
