
// Minerva Includes
#include <minerva/matrix/interface/FloatIterator.h>

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
	
	for(auto i = 0, end = offset.size(); i < end; ++i)
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

FloatIterator::FloatIterator(const Precision& p, const Dimension& s, void* d)
: _data(d), _stride(s), _precision(p)
{

}

FloatIterator::FloatIterator(const FloatIterator& ) = default;

FloatIterator& FloatIterator::operator=(const FloatIterator&) = default;

FloatReference FloatIterator::operator*()
{
	return FloatReference(_precision, detail::getAddress(_data, ));
}

FloatIterator FloatIterator::operator++()
{
	return FloatIterator(_precision, _size, _stride, detail::advance(_offset, _size), _data);
}

bool FloatIterator::operator==(const FloatIterator& i)
{
	return (_data == i._data) &&
		   (_precision == i._precision) &&
		   (_offset == i._offset) &&
		   (_size == i._size) &&
		   (_stride == i._stride);
}

bool FloatIterator::operator==(const ConstFloatIterator&);

bool FloatIterator::operator!=(const FloatIterator& i)
{
	return !(*this == i);
}

bool FloatIterator::operator!=(const ConstFloatIterator& i)
{
	return !(*this == i);
}

ConstFloatIterator::ConstFloatIterator();
ConstFloatIterator::ConstFloatIterator(const Precision& p, const Dimension& s, const void* d);

ConstFloatIterator::ConstFloatIterator(const FloatIterator& );
ConstFloatIterator::ConstFloatIterator(const ConstFloatIterator& );

ConstFloatIterator& ConstFloatIterator::operator=(const FloatIterator&);
ConstFloatIterator& ConstFloatIterator::operator=(const ConstFloatIterator&);

ConstFloatReference ConstFloatIterator::operator*();

ConstFloatIterator ConstFloatIterator::operator++();

bool ConstFloatIterator::operator==(const FloatIterator&);
bool ConstFloatIterator::operator==(const ConstFloatIterator&);

bool ConstFloatIterator::operator!=(const FloatIterator&);
bool ConstFloatIterator::operator!=(const ConstFloatIterator&);

}
}




