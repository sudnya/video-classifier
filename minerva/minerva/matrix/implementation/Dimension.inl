
#pragma once

// Minerva Includes
#include <minerva/matrix/interface/Dimension.h>

// Standard Library Includes
#include <sstream>

namespace minerva
{
namespace matrix
{

CUDA_DECORATOR Dimension::Dimension()
: _arity(0)
{
}

CUDA_DECORATOR Dimension::Dimension(std::initializer_list<size_t> sizes)
: _arity(sizes.size())
{
    auto element = sizes.begin();
    for(size_t i = 0; i < size(); ++i, ++element)
    {
        _storage[i] = *element;
    }
}

CUDA_DECORATOR void Dimension::push_back(size_t size)
{
    assert(_arity < capacity);
    _storage[_arity++] = size;
}

CUDA_DECORATOR void Dimension::pop_back()
{
    pop_back(1);
}

CUDA_DECORATOR void Dimension::pop_back(size_t size)
{
    assert(_arity >= size);

    _arity -= size;
}

CUDA_DECORATOR size_t Dimension::size() const
{
    return _arity;
}

CUDA_DECORATOR bool Dimension::empty() const
{
    return size() == 0;
}

CUDA_DECORATOR size_t Dimension::product() const
{
    size_t size = 1;

    for(auto element : *this)
    {
        size *= element;
    }

    return size;
}

CUDA_DECORATOR size_t& Dimension::back()
{
    assert(!empty());

    return _storage[size() - 1];
}

CUDA_DECORATOR size_t Dimension::back() const
{
    assert(!empty());

    return _storage[size() - 1];
}

CUDA_DECORATOR size_t& Dimension::front()
{
    return *begin();
}

CUDA_DECORATOR size_t Dimension::front() const
{
    return *begin();
}

CUDA_DECORATOR Dimension::iterator Dimension::begin()
{
    return _storage;
}

CUDA_DECORATOR Dimension::const_iterator Dimension::begin() const
{
    return _storage;
}

CUDA_DECORATOR Dimension::iterator Dimension::end()
{
    return begin() + size();
}

CUDA_DECORATOR Dimension::const_iterator Dimension::end() const
{
    return begin() + size();
}

CUDA_DECORATOR size_t Dimension::operator[](size_t position) const
{
    assert(position < size());

    return _storage[position];
}

CUDA_DECORATOR size_t& Dimension::operator[](size_t position)
{
    assert(position < size());

    return _storage[position];
}

std::string Dimension::toString() const
{
    std::stringstream stream;

    for(auto element : *this)
    {
        if(!stream.str().empty())
        {
            stream << ", ";
        }

        stream << element;
    }

    return stream.str();
}

CUDA_DECORATOR Dimension Dimension::operator+(const Dimension& d) const
{
    assert(size() == d.size());

    Dimension result;

    for(auto left = begin(), right = d.begin(); left != end(); ++left, ++right)
    {
        result.push_back(*left + *right);
    }

    return result;
}

CUDA_DECORATOR Dimension Dimension::operator-(const Dimension& d) const
{
    assert(size() == d.size());

    Dimension result;

    for(auto left = begin(), right = d.begin(); left != end(); ++left, ++right)
    {
        result.push_back(*left - *right);
    }

    return result;
}

CUDA_DECORATOR Dimension Dimension::operator/(const Dimension& d) const
{
    assert(size() == d.size());

    Dimension result;

    for(auto left = begin(), right = d.begin(); left != end(); ++left, ++right)
    {
        result.push_back(*left / *right);
    }

    return result;
}

CUDA_DECORATOR Dimension Dimension::operator*(const Dimension& d) const
{
    assert(size() == d.size());

    Dimension result;

    for(auto left = begin(), right = d.begin(); left != end(); ++left, ++right)
    {
        result.push_back(*left * *right);
    }

    return result;
}

CUDA_DECORATOR bool Dimension::operator==(const Dimension& d) const
{
    if(d.size() != size())
    {
        return false;
    }

    for(auto l = begin(), r = d.begin(); l != end(); ++l, ++r)
    {
        if(*l != *r)
        {
            return false;
        }
    }

    return true;
}

CUDA_DECORATOR bool Dimension::operator!=(const Dimension& d) const
{
    return !(d == *this);
}

}
}


