
#pragma once

// Minerva Includes
#include <minerva/matrix/interface/FloatReference.h>

namespace minerva
{
namespace matrix
{

class ConstFloatIterator;

class FloatIterator
{
public:
	FloatReference operator*();

public:
	FloatIterator operator++();

public:
	bool operator==(const FloatIterator&);
	bool operator==(const ConstFloatIterator&);

public:
	bool operator!=(const FloatIterator&);
	bool operator!=(const ConstFloatIterator&);

private:
	void*     _data;
	Precision _precision;

};

class ConstFloatIterator
{
public:
	ConstFloatReference operator*();

public:
	ConstFloatIterator operator++();

public:
	bool operator==(const FloatIterator&);
	bool operator==(const ConstFloatIterator&);

public:
	bool operator!=(const FloatIterator&);
	bool operator!=(const ConstFloatIterator&);

private:
	const void* _data;
	Precision   _precision;

};

}
}



