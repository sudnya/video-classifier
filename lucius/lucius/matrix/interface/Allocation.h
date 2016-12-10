

#pragma once

// Standard Library Includes
#include <cstdint>
#include <cstddef>

namespace lucius
{
namespace matrix
{

class Allocation
{
public:
    typedef uint8_t* pointer;
    typedef const pointer const_pointer;

public:
    Allocation();
    Allocation(size_t size);

public:
    Allocation(Allocation&& a);
    Allocation& operator=(Allocation&& a);

public:
    ~Allocation();

public:
    void clear();

public:
          pointer data();
    const_pointer data() const;

public:
    size_t size() const;

private:
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;

private:
    pointer _begin;
    pointer _end;

};

}
}



