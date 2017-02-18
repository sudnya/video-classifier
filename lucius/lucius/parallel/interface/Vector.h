#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/ScalarOperations.h>
#include <lucius/parallel/interface/assert.h>

namespace lucius
{
namespace parallel
{

/*! \brief A CUDA compatible list class with C++ standard library syntax and semantics. */
template <typename T>
class vector
{
public:
    typedef T& reference;
    typedef const T& const_reference;
    typedef pointer_iterator<T> iterator;
    typedef const_pointer_iterator<T> const_iterator;
    typedef parallel::reverse_iterator<iterator> reverse_iterator;
    typedef parallel::reverse_iterator<const_iterator> const_reverse_iterator;

public:
    CUDA_DECORATOR vector()
    : vector(0)
    {

    }

    CUDA_DECORATOR explicit vector(size_t count)
    : _begin(new T[_align(count)]), _size(count), _capacity(_align(count))
    {

    }

    CUDA_DECORATOR ~vector()
    {
        delete[] _begin;
    }

    CUDA_DECORATOR vector(const vector& s)
    : vector()
    {
        insert(end(), s.begin(), s.end());
    }

    CUDA_DECORATOR vector(vector&& s)
    : vector()
    {
        swap(s);
    }

public:
    CUDA_DECORATOR vector& operator=(const vector& v)
    {
        if(&v == this)
        {
            return *this;
        }

        return *this;
    }

    CUDA_DECORATOR vector& operator=(vector&& v)
    {
        if(&v == this)
        {
            return *this;
        }

        clear();

        swap(v);

        return *this;
    }

public:
    CUDA_DECORATOR size_t size() const
    {
        return _size;
    }

    CUDA_DECORATOR bool empty() const
    {
        return _size == 0;
    }

    CUDA_DECORATOR size_t capacity() const
    {
        return _capacity;
    }

public:
    CUDA_DECORATOR void clear() const
    {
        resize(0);
    }

    CUDA_DECORATOR void resize(size_t size)
    {
        if(size > capacity())
        {
            reserve(_align(size));
        }

        _size = size;
    }

    CUDA_DECORATOR void reserve(size_t newCapacity)
    {
        if(newCapacity <= capacity())
        {
            return;
        }

        size_t alignedCapacity = _align(newCapacity);

        T* data = new T[alignedCapacity];

        parallel::copy(begin(), end(), data);

        delete[] _begin;

        _begin    = data;
        _capacity = alignedCapacity;
    }

public:
    CUDA_DECORATOR void push_back(const T& v)
    {
        insert(end(), v);
    }

    CUDA_DECORATOR iterator insert(iterator position, const T& v)
    {
        return insert(position, const_iterator(&v), const_iterator(&v + 1));
    }

    template<typename Iterator>
    CUDA_DECORATOR iterator insert(iterator position, Iterator beginRange, Iterator endRange)
    {
        size_t newElements = parallel::distance(beginRange, endRange);
        size_t newSize = size() + newElements;

        if(newSize > capacity())
        {
            size_t offset = parallel::distance(begin(), position);
            reserve(newSize);
            position = begin() + offset;
        }

        parallel::copy_backward(position, end(), position + newElements);
        parallel::copy(beginRange, endRange, position);

        _size += newElements;

        return position;
    }

public:
    CUDA_DECORATOR reference operator[](size_t index)
    {
        return _begin[index];
    }

    CUDA_DECORATOR const_reference operator[](size_t index) const
    {
        return _begin[index];
    }

public:
    CUDA_DECORATOR iterator begin()
    {
        return _begin;
    }

    CUDA_DECORATOR iterator end()
    {
        return _begin + size();
    }

    CUDA_DECORATOR const_iterator begin() const
    {
        return _begin;
    }

    CUDA_DECORATOR const_iterator end() const
    {
        return _begin + size();
    }

public:
    CUDA_DECORATOR reverse_iterator rbegin()
    {
        return parallel::make_reverse(end());
    }

    CUDA_DECORATOR reverse_iterator rend()
    {
        return parallel::make_reverse(begin());
    }

    CUDA_DECORATOR const_reverse_iterator rbegin() const
    {
        return parallel::make_reverse(end());
    }

    CUDA_DECORATOR const_reverse_iterator rend() const
    {
        return parallel::make_reverse(begin());
    }

public:
    CUDA_DECORATOR vector& swap(vector&& s)
    {
        std::swap(_begin,    s._begin);
        std::swap(_size,     s._size);
        std::swap(_capacity, s._capacity);

        return *this;
    }

private:
    static constexpr size_t PageBytes    = 4 * 8;
    static constexpr size_t PageElements = (PageBytes + sizeof(T) - 1) / sizeof(T);

private:
    CUDA_DECORATOR static size_t _align(size_t s)
    {
        size_t remainder = s % PageElements;

        return remainder == 0 ? s : s + PageElements - remainder;
    }

public:
    T*     _begin;
    size_t _size;
    size_t _capacity;

};

} // namespace parallel
} // namespace lucius

