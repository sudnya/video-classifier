
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/cstring.h>
#include <lucius/parallel/interface/ScalarOperations.h>

namespace lucius
{
namespace parallel
{

/*! \brief A CUDA compatible string class with C++ standard library syntax and semantics. */
class string
{
public:
    CUDA_DECORATOR string()
    {
        _reset();
    }

    CUDA_DECORATOR string(const char* s)
    : string()
    {
        resize(parallel::strlen(s));

        parallel::memcpy(data(), s, capacity());
    }

    CUDA_DECORATOR string(const string& s)
    {
        resize(s.size());

        parallel::memcpy(data(), s.data(), s.size());
    }

    CUDA_DECORATOR ~string()
    {
        _delete();
    }

    CUDA_DECORATOR string(string&& s)
    : _data(reinterpret_cast<char*>(s.data())), _size(s.size())
    {
        s._reset();
    }

public:
    CUDA_DECORATOR string& operator=(const string& s)
    {
        if(this == &s)
        {
            return *this;
        }

        resize(s.capacity());

        parallel::memcpy(data(), s.data(), s.capacity());

        return *this;
    }

    CUDA_DECORATOR string& operator=(string&& s)
    {
        if(this == &s)
        {
            return *this;
        }

        parallel::swap(s._data, _data);
        parallel::swap(s._size, _size);

        s.clear();

        return *this;
    }

public:
    CUDA_DECORATOR void clear()
    {
        _delete();
        _reset();
    }

    CUDA_DECORATOR void resize(size_t newSize)
    {
        char* newData = new char[newSize + 1];

        size_t copiedSize = parallel::min(newSize, size());

        parallel::memcpy(newData, data(), copiedSize);

        _delete();

        _data = newData;
        _size = newSize;

        _zeroEnd();
    }

    CUDA_DECORATOR void push_back(char c)
    {
        resize(size() + 1);

        back() = c;
    }

public:
    CUDA_DECORATOR const char* c_str() const
    {
        return _data;
    }

    CUDA_DECORATOR const void* data() const
    {
        return _data;
    }

    CUDA_DECORATOR void* data()
    {
        return _data;
    }

    CUDA_DECORATOR size_t size() const
    {
        return _size;
    }

    CUDA_DECORATOR size_t capacity() const
    {
        return _size + 1;
    }

public:
    CUDA_DECORATOR char& back()
    {
        return _data[size()-1];
    }

    CUDA_DECORATOR char& front()
    {
        return _data[0];
    }

public:
    typedef char* iterator;
    typedef const char* const_iterator;

public:
    CUDA_DECORATOR iterator begin()
    {
        return _data;
    }

    CUDA_DECORATOR const_iterator begin() const
    {
        return _data;
    }

    CUDA_DECORATOR iterator end()
    {
        return begin() + size();
    }

    CUDA_DECORATOR const_iterator end() const
    {
        return begin() + size();
    }

private:
    CUDA_DECORATOR void _zeroEnd()
    {
        _data[size()] = '\0';
    }

    CUDA_DECORATOR void _delete()
    {
        delete[] _data;
    }

    CUDA_DECORATOR void _reset()
    {
        _data = new char[1];
        _size = 0;
        _zeroEnd();
    }

private:
    char* _data;
    size_t _size;

};

CUDA_DECORATOR inline string operator+(const string& lhs, const string& rhs)
{
    auto result = lhs;

    for(auto& i : rhs)
    {
        result.push_back(i);
    }

    return result;
}

CUDA_DECORATOR inline string operator+(const string& lhs, const char* rhs)
{
    return lhs + string(rhs);
}

CUDA_DECORATOR inline string operator+(const char* lhs, const string& rhs)
{
    return string(lhs) + rhs;
}

}
}




