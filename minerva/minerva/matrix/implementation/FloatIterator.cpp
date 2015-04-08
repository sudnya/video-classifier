
// Minerva Includes
#include <minerva/matrix/interface/FloatIterator.h>
#include <minerva/matrix/interface/MatrixTransformations.h>

namespace minerva
{
namespace matrix
{

namespace detail
{

static Dimension advance(const Dimension& offset, const Dimension& size)
{
	Dimension result;
	
	bool carry = true;
	
	for(size_t i = 0, end = offset.size(); i < end; ++i)
	{
		if(carry)
		{
			if(offset[i] + 1 >= size[i])
			{
				result.push_back(0);
			}
			else
			{
				result.push_back(offset[i] + 1);
				carry = false;
			}
		}
		else
		{
			result.push_back(offset[i]);
		}
	}
	
	return result;
}

}

FloatIterator::FloatIterator() = default;

FloatIterator::FloatIterator(const Precision& p, const Dimension& size,
	const Dimension& stride, const Dimension& offset, void* d)
: _data(d), _stride(stride), _size(size), _offset(offset), _precision(p)
{

}

FloatIterator::FloatIterator(const FloatIterator& ) = default;

FloatIterator& FloatIterator::operator=(const FloatIterator&) = default;

FloatReference FloatIterator::operator*()
{
	return FloatReference(_precision, getAddress(_stride, _offset, _data, _precision));
}

FloatIterator FloatIterator::operator++()
{
	return FloatIterator(_precision, _size, _stride, detail::advance(_offset, _size), _data);
}

bool FloatIterator::operator==(const FloatIterator& i) const
{
	return (_data == i._data) &&
		   (_precision == i._precision) &&
		   (_offset == i._offset) &&
		   (_size == i._size) &&
		   (_stride == i._stride);
}

bool FloatIterator::operator==(const ConstFloatIterator& i) const
{
	return (_data == i._data) &&
		   (_precision == i._precision) &&
		   (_offset == i._offset) &&
		   (_size == i._size) &&
		   (_stride == i._stride);
}

bool FloatIterator::operator!=(const FloatIterator& i) const
{
	return !(*this == i);
}

bool FloatIterator::operator!=(const ConstFloatIterator& i) const
{
	return !(*this == i);
}

ConstFloatIterator::ConstFloatIterator() = default;

ConstFloatIterator::ConstFloatIterator(const Precision& p, const Dimension& size,
	const Dimension& stride, const Dimension& offset, const void* d)
: _data(d), _stride(stride), _size(size), _offset(offset), _precision(p)
{

}

ConstFloatIterator::ConstFloatIterator(const FloatIterator& i)
: _data(i._data), _stride(i._stride), _size(i._size), _offset(i._offset), _precision(i._precision)
{

}

ConstFloatIterator::ConstFloatIterator(const ConstFloatIterator& ) = default;

ConstFloatIterator& ConstFloatIterator::operator=(const FloatIterator& i)
{
	_data      = i._data;
	_stride    = i._stride;
	_size      = i._size;
	_offset    = i._offset;
	_precision = i._precision;

	return *this;
}

ConstFloatIterator& ConstFloatIterator::operator=(const ConstFloatIterator&) = default;

ConstFloatReference ConstFloatIterator::operator*()
{
	return ConstFloatReference(_precision, getAddress(_stride, _offset, _data, _precision));
}

ConstFloatIterator ConstFloatIterator::operator++()
{
	return ConstFloatIterator(_precision, _size, _stride, detail::advance(_offset, _size), _data);
}

bool ConstFloatIterator::operator==(const FloatIterator& i) const
{
	return (_data == i._data) &&
		   (_precision == i._precision) &&
		   (_offset == i._offset) &&
		   (_size == i._size) &&
		   (_stride == i._stride);
}

bool ConstFloatIterator::operator==(const ConstFloatIterator& i) const
{
	return (_data == i._data) &&
		   (_precision == i._precision) &&
		   (_offset == i._offset) &&
		   (_size == i._size) &&
		   (_stride == i._stride);
}

bool ConstFloatIterator::operator!=(const FloatIterator& i) const
{
	return !(*this == i);
}

bool ConstFloatIterator::operator!=(const ConstFloatIterator& i) const
{
	return !(*this == i);
}

}
}




