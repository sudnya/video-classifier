

#pragma once

// Standard Library Includes
#include <cstdint>

namespace minerva
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
	      pointer data();
	const_pointer data() const;

private:
	typedef std::shared_ptr<uint8_t> shared_pointer;

private:
	shared_pointer _begin;
	       pointer _end;

};

}
}



