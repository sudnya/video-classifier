

#pragma once

#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/Iterator.h>

#include <cstddef>
#include <algorithm>

namespace lucius
{
namespace parallel
{

template <typename T>
CUDA_DECORATOR inline T min(const T& left, const T& right)
{
    #ifdef __NVCC__
    return ::min(left, right);
    #else
    return std::min(left, right);
    #endif
}

CUDA_DECORATOR inline size_t min(const size_t& left, const size_t& right)
{
    #ifdef __NVCC__
    return ::min((unsigned long long)left, (unsigned long long)right);
    #else
    return std::min(left, right);
    #endif
}

template <typename T>
CUDA_DECORATOR inline T max(const T& left, const T& right)
{
    #ifdef __NVCC__
    return ::max(left, right);
    #else
    return std::max(left, right);
    #endif
}

CUDA_DECORATOR inline size_t max(const size_t& left, const size_t& right)
{
    #ifdef __NVCC__
    return ::max((unsigned long long)left, (unsigned long long)right);
    #else
    return std::max(left, right);
    #endif
}

template <typename T>
CUDA_DECORATOR inline void swap(T& left, T& right)
{
    #ifdef __NVCC__
    T temp = std::move(left);
    left = std::move(right);
    right = std::move(temp);
    #else
    std::swap(left, right);
    #endif
}

template <typename InputIterator, typename OutputIterator>
CUDA_DECORATOR inline OutputIterator copy(InputIterator begin, InputIterator end,
    OutputIterator result)
{
    for(auto i = begin; i != end; ++i)
    {
        *(result++) = i;
    }

    return result;
}

template <typename InputIterator, typename OutputIterator>
CUDA_DECORATOR inline OutputIterator copy_backward(InputIterator begin, InputIterator end,
    OutputIterator result)
{
    return copy(make_reverse(end), make_reverse(begin), make_reverse(result));
}

template <typename Iterator>
CUDA_DECORATOR inline typename Iterator::diff_type distance(Iterator begin, Iterator end)
{
    return end - begin;
}

}
}


