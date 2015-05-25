
// Minerva Includes
#include <minerva/parallel/interface/ConcurrentCollectives.h>

namespace minerva
{
namespace parallel
{

CUDA_DECORATOR ThreadGroup::ThreadGroup(size_t size, size_t id)
: _size(size), _id(id)
{

}

CUDA_DECORATOR size_t ThreadGroup::size() const
{
	return _size;
}

CUDA_DECORATOR size_t ThreadGroup::id() const
{
	return _id;
}

}
}

