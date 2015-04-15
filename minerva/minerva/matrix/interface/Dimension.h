#pragma once

// Standard Library Includes
#include <array>
#include <cassert>
#include <string>

namespace minerva
{
namespace matrix
{

class Dimension
{
private:
	static constexpr size_t capacity = 9;

private:
	typedef std::array<size_t, capacity> Storage;

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
	
	Dimension();
	Dimension(std::initializer_list<size_t>);

public:
	void push_back(size_t );

	void pop_back();
	void pop_back(size_t );

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

public:
	Dimension operator-(const Dimension& ) const;
	Dimension operator+(const Dimension& ) const;
	Dimension operator/(const Dimension& ) const;
	Dimension operator*(const Dimension& ) const;

public:
	bool operator==(const Dimension& ) const;
	bool operator!=(const Dimension& ) const;

private:
	template<typename T>
	void fill(Storage& storage, size_t& arity, T argument)
	{
		assert(arity < capacity);
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

