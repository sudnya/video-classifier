
#pragma once

// Lucious Includes
#include <lucious/parallel/interface/ConcurrentCollectives.h>

namespace lucious
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

}
}

