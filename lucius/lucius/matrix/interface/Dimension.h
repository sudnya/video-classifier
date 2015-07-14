#pragma once

// Lucious Includes
#include <lucious/parallel/interface/cuda.h>
#include <lucious/parallel/interface/assert.h>

// Standard Library Includes
#include <string>

namespace lucious
{
namespace matrix
{

class Dimension
{
private:
    static constexpr size_t capacity = 9;

private:
    typedef size_t Storage[capacity];

public:
    typedef size_t*       iterator;
    typedef const size_t* const_iterator;

public:
    template<typename... Args>
    CUDA_DECORATOR inline Dimension(Args... args)
    : _arity(0)
    {
        fill(_storage, _arity, args...);
    }

    CUDA_DECORATOR inline Dimension();
    CUDA_DECORATOR inline Dimension(std::initializer_list<size_t>);

public:
    CUDA_DECORATOR inline void push_back(size_t );

    CUDA_DECORATOR inline void pop_back();
    CUDA_DECORATOR inline void pop_back(size_t );

public:
    CUDA_DECORATOR inline size_t size() const;
    CUDA_DECORATOR inline bool empty() const;

public:
    CUDA_DECORATOR inline size_t& back();
    CUDA_DECORATOR inline size_t  back() const;

    CUDA_DECORATOR inline size_t& front();
    CUDA_DECORATOR inline size_t  front() const;

public:
    CUDA_DECORATOR inline size_t product() const;

public:
    CUDA_DECORATOR inline iterator begin();
    CUDA_DECORATOR inline const_iterator begin() const;

    CUDA_DECORATOR inline iterator end();
    CUDA_DECORATOR inline const_iterator end() const;

public:
    CUDA_DECORATOR inline size_t  operator[](size_t position) const;
    CUDA_DECORATOR inline size_t& operator[](size_t position);

public:
    inline std::string toString() const;

public:
    static inline Dimension fromString(const std::string& );

public:
    CUDA_DECORATOR inline Dimension operator-(const Dimension& ) const;
    CUDA_DECORATOR inline Dimension operator+(const Dimension& ) const;
    CUDA_DECORATOR inline Dimension operator/(const Dimension& ) const;
    CUDA_DECORATOR inline Dimension operator*(const Dimension& ) const;

public:
    CUDA_DECORATOR inline bool operator==(const Dimension& ) const;
    CUDA_DECORATOR inline bool operator!=(const Dimension& ) const;

private:
    template<typename T>
    CUDA_DECORATOR inline void fill(Storage& storage, size_t& arity, T argument)
    {
        assert(arity < capacity);
        storage[arity++] = argument;
    }

    template<typename T, typename... Args>
    CUDA_DECORATOR inline void fill(Storage& storage, size_t& arity, T argument, Args... args)
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

#include <lucious/matrix/implementation/Dimension.inl>

