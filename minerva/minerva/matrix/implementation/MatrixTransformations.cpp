
// Minerva Includes
#include <minerva/matrix/interface/MatrixTransformations.h>

// Standard Library Includes
#include <set>

namespace minerva
{
namespace matrix
{

Dimension linearStride(const Dimension& size)
{
	Dimension stride;
	
	size_t step = 1;
	
	for (auto sizeStep : size)
	{
		stride.push_back(step);
		step *= sizeStep;
	}
	
	return stride;
}

Dimension zeros(const Dimension& size)
{
	Dimension result;
	
	for(size_t i = 0, arity = size.size(); i < arity; ++i)
	{
		result.push_back(0);
	}
	
	return result;
}

Dimension removeDimensions(const Dimension& base, const Dimension& toRemove)
{
	if(toRemove.size() == 0)
	{
		return Dimension({1});
	}
	
	std::set<size_t> removed;
	
	for(auto i : toRemove)
	{
		removed.insert(i);
	}
	
	Dimension result;
	
	for(auto i : toRemove)
	{
		if(removed.count(i) == 0)
		{
			result.push_back(i);
		}
	}
	
	return result;
}

static size_t getOffset(const Dimension& stride, const Dimension& position)
{
	size_t offset = 0;
	
	for(auto i = 0; i < stride.size(); ++i)
	{
		offset += stride[i] * position[i];
	}
	
	return offset;
}

void* getAddress(const Dimension& stride, const Dimension& position, void* data, const Precision& precision)
{
	size_t offset = getOffset(stride, position);
	
	uint8_t* address = static_cast<uint8_t*>(data);
	
	return address + precision.size() * offset;
}

const void* getAddress(const Dimension& stride, const Dimension& position, const void* data, const Precision& precision)
{
	size_t offset = getOffset(stride, position);
	
	const uint8_t* address = static_cast<const uint8_t*>(data);
	
	return address + precision.size() * offset;
}

}
}	


