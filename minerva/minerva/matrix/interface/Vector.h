/*	\file   Vector.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Vector class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <cstring>

namespace minerva
{

namespace matrix
{

class Vector
{
public:
	typedef std::vector<float> FloatVector;
	typedef FloatVector::iterator       iterator;
	typedef FloatVector::const_iterator const_iterator;

public:
	Vector(const Vector&);
	Vector(size_t size = 0);

public:
	iterator       begin();
	const_iterator begin() const;

	iterator       end();
	const_iterator end() const;

public:
	      float& operator[](size_t index);
	const float& operator[](size_t index) const;

public:
	size_t size() const;
	
public:
	void* data();
	const void* data() const;

private:
	FloatVector _data;

};

}

}


