
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/ConcurrentCollectives.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{
namespace parallel
{

CUDA_DECORATOR inline ThreadGroup::ThreadGroup(size_t size, size_t id)
: _size(size), _id(id)
{

}

CUDA_DECORATOR inline size_t ThreadGroup::size() const
{
	return _size;
}

CUDA_DECORATOR inline size_t ThreadGroup::id() const
{
	return _id;
}

CUDA_DECORATOR inline ThreadGroup partitionThreadGroup(ThreadGroup g, size_t subgroupSize)
{
    return ThreadGroup(subgroupSize, g.id() % subgroupSize);
}

CUDA_DECORATOR inline ThreadGroup partitionThreadGroupAtLevel(ThreadGroup g, size_t level)
{
    if(level == 0)
    {
        return partitionThreadGroup(g, 1);
    }
    else if(level == 1)
    {
        return partitionThreadGroup(g, 32);
    }
    else if(level == 2)
    {
        return partitionThreadGroup(g, 128);
    }

    return g;
}

CUDA_DECORATOR inline ThreadGroup getRelativeGroup(ThreadGroup inner, ThreadGroup outer)
{
    return ThreadGroup(outer.size() / inner.size(), outer.id() / inner.size());
}

CUDA_DECORATOR inline void barrier(ThreadGroup g)
{
    if(g.size() <= 32)
    {
        return;
    }
    else if(g.size() <= 128)
    {
        #ifdef __CUDA_ARCH__
        __syncthreads();
        #endif
        return;
    }

    assert(false && "Not implemented.");

}

template<typename T>
CUDA_DECORATOR inline T gather(ThreadGroup g, T value, size_t index)
{
    if(g.size() == 0)
    {
        return value;
    }
    else if(g.size() <= 32)
    {
        return __shfl(value, index, g.size());
    }
    else if(g.size() <= 128)
    {
        T result = value;
        #ifdef __CUDA_ARCH__
        __shared__ T data[128];
        data[g.id()] = value;
        barrier(g);

        if(index < 128)
        {
            result = data[index];
        }

        barrier(g);
        #endif

        return result;
    }

    assert(false && "Not implemented.");

    return value;
}

template<typename T, typename Function>
CUDA_DECORATOR inline T reduce(ThreadGroup g, T value, Function f)
{
    T result = value;

    for(size_t i = g.size() / 2; i >= 1; i /= 2)
    {
        result = f(result, gather(g, value, g.id() + i));
    }

    return result;
}

}
}

