/*  \file   IteratorRange.h
    \date   April 10, 2018
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the IteratorRange class.
*/

#pragma once

namespace lucius
{
namespace util
{

/*! \brief Shorthand for creating an iterator range for passing to a range based loop */
template <typename T>
class IteratorRange
{
public:
    IteratorRange(T begin, T end)
    : _begin(begin), _end(end)
    {

    }

public:
    T begin() const
    {
        return _begin;
    }

    T end() const
    {
        return _end;
    }

private:
    T _begin;
    T _end;


};

} // namespace util
} // namespace lucius

