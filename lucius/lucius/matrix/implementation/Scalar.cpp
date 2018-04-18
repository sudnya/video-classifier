/*  \file   Scalar.cpp
    \date   April 17, 2018
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the Scalar class.
*/

// Luicus Includes
#include <lucius/matrix/interface/Scalar.h>

// Standard Library Includes
#include <sstream>

namespace lucius
{
namespace matrix
{

Scalar::Scalar()
{

}

const Precision& Scalar::getPrecision() const
{
    return _precision;
}

std::string Scalar::toString() const
{
    std::stringstream stream;

    if(getPrecision() == SizeTPrecision())
    {
        stream << get<size_t>();
    }
    else if(getPrecision() == SinglePrecision())
    {
        stream << get<float>();
    }
    else if(getPrecision() == HalfPrecision())
    {
        stream << get<float>();
    }
    else if(getPrecision() == DoublePrecision())
    {
        stream << get<double>();
    }
    else
    {
        stream << "invalid";
    }

    return stream.str();
}

}
}


