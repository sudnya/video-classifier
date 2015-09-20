
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>

// Standard Library Includes
#include <cstddef>

namespace lucius
{
namespace parallel
{

class ThreadGroup
{
public:
    CUDA_DECORATOR inline ThreadGroup(size_t size, size_t id);

public:
    CUDA_DECORATOR inline size_t size() const;
    CUDA_DECORATOR inline size_t id()   const;

public:
    size_t _size;
    size_t _id;

};

}
}

#include <lucius/parallel/implementation/ConcurrentCollectives.inl>



