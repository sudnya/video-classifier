
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

template<size_t level>
class GroupLevelSize
{
public:
    CUDA_DECORATOR static constexpr size_t size()
    {
        #ifdef __CUDA_ARCH__
        return level == 0 ? 1  :
               ((level == 1) ? 32 : 512);
        #else
        return 1;
        #endif
    }
};

CUDA_DECORATOR inline ThreadGroup partitionThreadGroup(ThreadGroup g, size_t subgroupSize);
CUDA_DECORATOR inline ThreadGroup partitionThreadGroupAtLevel(ThreadGroup g, size_t level);

CUDA_DECORATOR inline ThreadGroup getRelativeGroup(ThreadGroup inner, ThreadGroup outer);

CUDA_DECORATOR inline void barrier(ThreadGroup g);

template<typename T>
CUDA_DECORATOR inline T gather(ThreadGroup g, T value, size_t index);

template<typename T, typename Function>
CUDA_DECORATOR inline T reduce(ThreadGroup g, T value, Function f);

}
}

#include <lucius/parallel/implementation/ConcurrentCollectives.inl>



