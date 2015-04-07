
#pragma once

// Minerva Includes
#include <minerva/matrix/interface/FloatReference.h>
#include <minerva/matrix/interface/Dimension.h>

namespace minerva
{
namespace matrix
{

class ConstFloatIterator;

class FloatIterator
{
public:
	FloatIterator();
	FloatIterator(const Precision& p, const Dimension& s, void* d);

public:
	FloatIterator(const FloatIterator& );

public:
	FloatIterator& operator=(const FloatIterator&);

public:
	FloatReference operator*();

public:
	FloatIterator operator++();

public:
	bool operator==(const FloatIterator&) const;
	bool operator==(const ConstFloatIterator&) const;

public:
	bool operator!=(const FloatIterator&) const;
	bool operator!=(const ConstFloatIterator&) const;

private:
	void*     _data;
	Dimension _stride;
	Dimension _size;
	Dimension _offset;
	Precision _precision;

};

class ConstFloatIterator
{
public:
	ConstFloatIterator();
	ConstFloatIterator(const Precision& p, const Dimension& s, const void* d);

public:
	ConstFloatIterator(const FloatIterator& );
	ConstFloatIterator(const ConstFloatIterator& );

public:
	ConstFloatIterator& operator=(const FloatIterator&);
	ConstFloatIterator& operator=(const ConstFloatIterator&);

public:
	ConstFloatReference operator*();

public:
	ConstFloatIterator operator++();

public:
	bool operator==(const FloatIterator&) const;
	bool operator==(const ConstFloatIterator&) const;

public:
	bool operator!=(const FloatIterator&) const;
	bool operator!=(const ConstFloatIterator&) const;

private:
	const void* _data;
	Dimension   _stride;
	Dimension   _size;
	Dimension   _offset;
	Precision   _precision;

};

}
}



