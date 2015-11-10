
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
    CUDA_DECORATOR inline ThreadGroup(int size, int id);

public:
    CUDA_DECORATOR inline int size() const;
    CUDA_DECORATOR inline int id()   const;

public:
    int _size;
    int _id;

};

template<int level>
class GroupLevelSize
{
public:
    CUDA_DECORATOR static constexpr int size()
    {
        #ifdef __CUDA_ARCH__
        return level == 0 ? 1  :
               ((level == 1) ? 32 : 512);
        #else
        return 1;
        #endif
    }
};

CUDA_DECORATOR inline ThreadGroup partitionThreadGroup(ThreadGroup g, int subgroupSize);
CUDA_DECORATOR inline ThreadGroup partitionThreadGroupAtLevel(ThreadGroup g, int level);

CUDA_DECORATOR inline ThreadGroup getRelativeGroup(ThreadGroup inner, ThreadGroup outer);

CUDA_DECORATOR inline void barrier(ThreadGroup g);

template<typename T>
CUDA_DECORATOR inline T gather(ThreadGroup g, T value, int index);

template<typename T, typename Function>
CUDA_DECORATOR inline T reduce(ThreadGroup g, T value, Function f);

}
}

#include <lucius/parallel/implementation/ConcurrentCollectives.inl>



