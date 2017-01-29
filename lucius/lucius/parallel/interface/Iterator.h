
#pragma once

// Standard Library Includes
#include <cstddef>

namespace lucius
{

namespace parallel
{

template <typename Iterator>
class reverse_iterator
{
public:
    typedef typename Iterator::value_type value_type;
    typedef typename Iterator::pointer_type pointer_type;
    typedef typename Iterator::reference_type reference_type;
    typedef typename Iterator::diff_type diff_type;

public:
    CUDA_DECORATOR reverse_iterator()
    {

    }

    CUDA_DECORATOR explicit reverse_iterator(Iterator iterator)
    : _iterator(iterator)
    {

    }

public:
    CUDA_DECORATOR bool operator==(const reverse_iterator& i)
    {
        return _iterator == i._iterator;
    }

    CUDA_DECORATOR bool operator!=(const reverse_iterator& i)
    {
        return _iterator != i._iterator;
    }

public:
    CUDA_DECORATOR reverse_iterator& operator++()
    {
        --_iterator;

        return *this;
    }

    CUDA_DECORATOR reverse_iterator operator++(int)
    {
        auto previous = *this;

        ++(*this);

        return previous;
    }

public:
    CUDA_DECORATOR reference_type operator*() const
    {
        return *(_iterator - 1);
    }

    CUDA_DECORATOR pointer_type operator->() const
    {
        return &(*(_iterator - 1));
    }

public:
    CUDA_DECORATOR Iterator get() const
    {
        return _iterator;
    }

private:
    Iterator _iterator;
};

template <typename Iterator>
CUDA_DECORATOR reverse_iterator<Iterator> make_reverse(Iterator i)
{
    return reverse_iterator<Iterator>(i);
}

template <typename T>
class pointer_iterator
{
public:
    typedef T value_type;
    typedef T* pointer_type;
    typedef T& reference_type;
    typedef std::ptrdiff_t diff_type;

public:
    CUDA_DECORATOR pointer_iterator()
    : pointer_iterator(nullptr)
    {

    }

    CUDA_DECORATOR pointer_iterator(pointer_type pointer)
    : _pointer(pointer)
    {

    }

public:
    CUDA_DECORATOR bool operator==(const pointer_iterator& i)
    {
        return _pointer == i._pointer;
    }

    CUDA_DECORATOR bool operator!=(const pointer_iterator& i)
    {
        return _pointer != i._pointer;
    }

public:
    CUDA_DECORATOR pointer_iterator& operator++()
    {
        ++_pointer;

        return *this;
    }

    CUDA_DECORATOR pointer_iterator operator++(int)
    {
        auto previous = *this;

        ++(*this);

        return previous;
    }

    CUDA_DECORATOR pointer_iterator& operator--()
    {
        --_pointer;

        return *this;
    }

    CUDA_DECORATOR pointer_iterator operator--(int)
    {
        auto previous = *this;

        --(*this);

        return previous;
    }

public:
    CUDA_DECORATOR diff_type operator-(const pointer_iterator& i) const
    {
        return _pointer - i._pointer;
    }

public:
    CUDA_DECORATOR pointer_iterator operator+(diff_type right) const
    {
        return pointer_iterator(_pointer + right);
    }

    CUDA_DECORATOR pointer_iterator operator-(diff_type right) const
    {
        return pointer_iterator(_pointer - right);
    }

public:
    CUDA_DECORATOR reference_type operator*() const
    {
        return *_pointer;
    }

    CUDA_DECORATOR pointer_type operator->() const
    {
        return _pointer;
    }


private:
    pointer_type _pointer;
};

template <typename T>
class const_pointer_iterator
{
public:
    typedef T value_type;
    typedef const T* pointer_type;
    typedef const T& reference_type;
    typedef std::ptrdiff_t diff_type;

public:
    CUDA_DECORATOR const_pointer_iterator()
    : const_pointer_iterator(nullptr)
    {

    }

    CUDA_DECORATOR const_pointer_iterator(pointer_type pointer)
    : _pointer(pointer)
    {

    }

public:
    CUDA_DECORATOR bool operator==(const const_pointer_iterator& i)
    {
        return _pointer == i._pointer;
    }

    CUDA_DECORATOR bool operator!=(const const_pointer_iterator& i)
    {
        return _pointer != i._pointer;
    }

public:
    CUDA_DECORATOR const_pointer_iterator& operator++()
    {
        ++_pointer;

        return *this;
    }

    CUDA_DECORATOR const_pointer_iterator operator++(int)
    {
        auto previous = *this;

        ++(*this);

        return previous;
    }

    CUDA_DECORATOR const_pointer_iterator& operator--()
    {
        --_pointer;

        return *this;
    }

    CUDA_DECORATOR const_pointer_iterator operator--(int)
    {
        auto previous = *this;

        --(*this);

        return previous;
    }

public:
    CUDA_DECORATOR diff_type operator-(const const_pointer_iterator& i) const
    {
        return _pointer - i._pointer;
    }

public:
    CUDA_DECORATOR const_pointer_iterator operator+(diff_type right) const
    {
        return const_pointer_iterator(_pointer + right);
    }

    CUDA_DECORATOR const_pointer_iterator operator-(diff_type right) const
    {
        return const_pointer_iterator(_pointer - right);
    }

public:
    CUDA_DECORATOR reference_type operator*() const
    {
        return *_pointer;
    }

    CUDA_DECORATOR pointer_type operator->() const
    {
        return _pointer;
    }


private:
    pointer_type _pointer;
};

}

}





