/*  \file   Scalar.h
    \date   April 17, 2018
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the Scalar class.
*/

#pragma once

// Lucius Includes
#include <lucius/matrix/interface/Precision.h>

// Standard Library Includes
#include <string>

namespace lucius
{
namespace matrix
{

/*! \brief An interface to operations on general purpose scalar data.

    TODO: Implement this with variant when support on OSX is better.
*/
class Scalar
{
public:
    Scalar();

    template <typename T>
    explicit Scalar(const T& );

public:
    template <typename T>
    T get() const;

    template <typename T>
    void set(const T& );

public:
    std::string toString() const;

public:
    const Precision& getPrecision() const;

private:
    union
    {
        float  _half;
        float  _float;
        double _double;
        size_t _size_t;
    };

public:
    Precision _precision;
};

}
}

#include <lucius/matrix/implementation/Scalar-inl.h>

