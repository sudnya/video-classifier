
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/List.h>

namespace lucius
{
namespace parallel
{

/*! \brief A CUDA compatible set class with C++ standard library syntax and semantics.

    TODO: implement this with a red black tree instead of a list.
*/
template <typename T>
class set
{
public:
    typedef typename list<T>::iterator iterator;
    typedef typename list<T>::const_iterator const_iterator;

public:
    CUDA_DECORATOR set()
    {
        // intentionall blank
    }

    CUDA_DECORATOR set(const set<T>& s)
    : _values(s._values)
    {

    }

    CUDA_DECORATOR ~set()
    {
        // intentionally blank
    }

public:
    CUDA_DECORATOR set<T>& operator=(const set<T>& s)
    {
        if(&s == this)
        {
            return *this;
        }

        _values = s._values;

        return *this;
    }

public:
    CUDA_DECORATOR size_t count(const T& key) const
    {
        size_t result = 0;

        for(auto& i : *this)
        {
            if(i == key)
            {
                ++result;
            }
        }

        return result;
    }

    CUDA_DECORATOR bool empty() const
    {
        return _values.empty();
    }

    CUDA_DECORATOR size_t size() const
    {
        return _values.size();
    }

public:
    CUDA_DECORATOR void clear()
    {
        return _values.clear();
    }

public:
    CUDA_DECORATOR iterator insert(const T& value)
    {
        auto existing = find(value);

        if(existing != end())
        {
            return existing;
        }

        return _values.insert(end(), value);
    }

    CUDA_DECORATOR iterator find(const T& key)
    {
        iterator position = begin();

        for(; position != end(); ++position)
        {
            if(*position == key)
            {
                break;
            }
        }

        return position;
    }

    CUDA_DECORATOR const_iterator find(const T& key) const
    {
        const_iterator position = begin();

        for(; position != end(); ++position)
        {
            if(*position == key)
            {
                break;
            }
        }

        return position;
    }

public:
    CUDA_DECORATOR iterator begin()
    {
        return _values.begin();
    }

    CUDA_DECORATOR const_iterator begin() const
    {
        return _values.begin();
    }

    CUDA_DECORATOR iterator end()
    {
        return _values.end();
    }

    CUDA_DECORATOR const_iterator end() const
    {
        return _values.end();
    }

private:
    list<T> _values;
};

} // namespace parallel

} // namespace lucius

