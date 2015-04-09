
// Minerva Includes
#include <minerva/matrix/interface/Allocation.h>
#include <minerva/parallel/interface/Memory.h>

namespace minerva
{
namespace matrix
{

Allocation::Allocation()
: _begin(nullptr), _end(nullptr)
{

}

Allocation::Allocation(size_t size)
: Allocation()
{
	_begin = static_cast<uint8_t*>(parallel::malloc(size));
	_end   = _begin + size;
}

Allocation::~Allocation()
{
	parallel::free(_begin);
}

Allocation::pointer Allocation::data()
{
	return _begin;
}

Allocation::const_pointer Allocation::data() const
{
	return _begin;
}

}
}



