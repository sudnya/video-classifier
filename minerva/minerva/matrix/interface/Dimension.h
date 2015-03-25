#pragma once

// Standard Library Includes
#include <array>
#include <cassert>

namespace minerva
{
namespace matrix
{

class Dimension
{
private:
	typedef std::array<size_t, 9> Storage;

public:
	typedef Storage::iterator       iterator;
	typedef Storage::const_iterator const_iterator;

public:
	template<typename... Args>
	Dimension(Args... args)
	: _arity(0)
	{
		fill(_storage, _arity, args...);
	}

	Dimension(std::initializer_list<size_t>);
	Dimension(const Dimension& );

public:
	void push_back(size_t );

public:
	size_t size() const;
	bool empty() const;

public:
	size_t product() const;

public:
	iterator begin();
	const_iterator begin() const;

	iterator end();
	const_iterator end() const;

public:
	size_t  operator[](size_t position) const;
	size_t& operator[](size_t position);

public:
	std::string toString() const;

private:
	template<typename T>
	void fill(Storage& storage, size_t& arity, T argument)
	{
		assert(arity < 9);
		storage[arity++] = argument;
	}
	
	template<typename T, typename... Args>
	void fill(Storage& storage, size_t& arity, T argument, Args... args)
	{
		fill(storage, arity, argument);
		fill(storage, arity, args...);
	}

private:
	Storage _storage;
	size_t  _arity;

};

}
}

