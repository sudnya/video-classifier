
// Lucius Includes
#include <lucius/matrix/interface/Allocation.h>
#include <lucius/parallel/interface/Memory.h>

// Standard Library Includes
#include <algorithm>

namespace lucius
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

Allocation::Allocation(Allocation&& allocation)
: _begin(allocation.data()), _end(allocation.data() + allocation.size())
{
    allocation._begin = nullptr;
    allocation._end   = nullptr;
}

Allocation& Allocation::operator=(Allocation&& allocation)
{
    if(this == &allocation)
    {
        return *this;
    }

    clear();

    std::swap(_begin, allocation._begin);
    std::swap(_end,   allocation._end);

    return *this;
}

Allocation::~Allocation()
{
    parallel::free(_begin);
}

void Allocation::clear()
{
    parallel::free(_begin);

    _begin = nullptr;
    _end   = nullptr;
}

Allocation::pointer Allocation::data()
{
    return _begin;
}

Allocation::const_pointer Allocation::data() const
{
    return _begin;
}

size_t Allocation::size() const
{
    return _end - _begin;
}

}
}




