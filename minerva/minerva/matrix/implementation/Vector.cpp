/*	\file   Vector.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Vector class.
*/

// Minerva Includes
#include <minerva/matrix/interface/Vector.h>

namespace minerva
{

namespace matrix
{

Vector::Vector(const Vector& v)
: _data(v._data)
{

}

Vector::Vector(size_t size)
: _data(size)
{

}

Vector::iterator Vector::begin()
{
	return _data.begin();
}

Vector::const_iterator Vector::begin() const
{
	return _data.begin();
}

Vector::iterator Vector::end()
{
	return _data.end();
}

Vector::const_iterator Vector::end() const
{
	return _data.end();
}

float& Vector::operator[](size_t index)
{
	return _data[index];
}

const float& Vector::operator[](size_t index) const
{
	return _data[index];
}

size_t Vector::size() const
{
	return _data.size();
}

void* Vector::data()
{
	return _data.data();
}

const void* Vector::data() const
{
	return _data.data();
}

}

}

