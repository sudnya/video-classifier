/*  \file   Scalar-inl.h
    \date   April 17, 2018
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The template implementation of the Scalar class.
*/

#pragma once

// Lucius Includes
#include <lucius/matrix/interface/Scalar.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace matrix
{

template <typename T>
Scalar::Scalar(const T& value)
{
    set(value);
}

template <typename T>
T Scalar::get() const
{
    assertM(false, "Not implemented");
}

template <>
inline size_t Scalar::get<size_t>() const
{
    assert(getPrecision() == SizeTPrecision());

    return _size_t;
}

template <>
inline float Scalar::get<float>() const
{
    assert(getPrecision() == SinglePrecision());

    return _float;
}

template <>
inline double Scalar::get<double>() const
{
    assert(getPrecision() == DoublePrecision());

    return _double;
}

template <typename T>
void Scalar::set(const T& )
{
    assertM(false, "Not implemented");
}

template <>
inline void Scalar::set<size_t>(const size_t& value)
{
    _size_t    = value;
    _precision = SizeTPrecision();
}

template <>
inline void Scalar::set<float>(const float& value)
{
    _float     = value;
    _precision = SinglePrecision();
}

template <>
inline void Scalar::set<double>(const double& value)
{
    _double    = value;
    _precision = DoublePrecision();
}

}
}


