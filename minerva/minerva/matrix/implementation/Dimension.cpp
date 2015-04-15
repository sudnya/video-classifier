
// Minerva Includes
#include <minerva/matrix/interface/Dimension.h>

// Standard Library Includes
#include <sstream>

namespace minerva
{
namespace matrix
{

Dimension::Dimension()
: _arity(0)
{
}

Dimension::Dimension(std::initializer_list<size_t> sizes)
: _arity(sizes.size())
{
    auto element = sizes.begin();
    for(size_t i = 0; i < size(); ++i, ++element)
    {
        _storage[i] = *element;
    }
}

void Dimension::push_back(size_t size)
{
    assert(_arity < capacity);
    _storage[_arity++] = size;
}

void Dimension::pop_back()
{
    pop_back(1);
}

void Dimension::pop_back(size_t size)
{
    assert(_arity >= size);
    
    _arity -= size;
}

size_t Dimension::size() const
{
    return _arity;
}

bool Dimension::empty() const
{
    return size() == 0;
}

size_t Dimension::product() const
{
    size_t size = 1;
    
    for(auto element : *this)
    {
        size *= element;
    }
    
    return size;
}

Dimension::iterator Dimension::begin()
{
    return _storage.begin();
}

Dimension::const_iterator Dimension::begin() const
{
    return _storage.begin();
}

Dimension::iterator Dimension::end()
{
    return begin() + size();
}

Dimension::const_iterator Dimension::end() const
{
    return begin() + size();
}

size_t Dimension::operator[](size_t position) const
{
    assert(position < size());
    
    return _storage[position];
}

size_t& Dimension::operator[](size_t position)
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
    
Dimension Dimension::operator+(const Dimension& d) const
{
    assert(size() == d.size());

    Dimension result;
    
    for(auto left = begin(), right = d.begin(); left != end(); ++left, ++right)
    {
        result.push_back(*left + *right);
    }
    
    return result;
}
    
Dimension Dimension::operator-(const Dimension& d) const
{
    assert(size() == d.size());

    Dimension result;
    
    for(auto left = begin(), right = d.begin(); left != end(); ++left, ++right)
    {
        result.push_back(*left - *right);
    }
    
    return result;
}

Dimension Dimension::operator/(const Dimension& d) const
{
    assert(size() == d.size());

    Dimension result;
    
    for(auto left = begin(), right = d.begin(); left != end(); ++left, ++right)
    {
        result.push_back(*left / *right);
    }
    
    return result;
}

Dimension Dimension::operator*(const Dimension& d) const
{
    assert(size() == d.size());

    Dimension result;
    
    for(auto left = begin(), right = d.begin(); left != end(); ++left, ++right)
    {
        result.push_back(*left * *right);
    }
    
    return result;
}

bool Dimension::operator==(const Dimension& d) const
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

bool Dimension::operator!=(const Dimension& d) const
{
    return !(d == *this);
}

}
}


