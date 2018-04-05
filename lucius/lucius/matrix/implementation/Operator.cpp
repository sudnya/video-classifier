/*  \file   Operator.cpp
    \date   February 19, 2018
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the Operator classes.
*/

// Lucius Includes
#include <lucius/matrix/interface/Operator.h>

#include <lucius/matrix/interface/StaticOperator.h>

namespace lucius
{

namespace matrix
{

Operator::Operator()
{

}

Operator::~Operator()
{
    // intentionally blank
}

Operator::Operator(const Operator& o)
: _operator(o.getStaticOperator().clone())
{

}

Operator::Operator(const StaticOperator& o)
: _operator(o.clone())
{

}

Operator& Operator::operator=(const Operator& o)
{
    if(this == &o)
    {
        return *this;
    }

    _operator = o.getStaticOperator().clone();

    return *this;
}

Operator& Operator::operator=(const StaticOperator& s)
{
    _operator = s.clone();

    return *this;
}

const StaticOperator& Operator::getStaticOperator() const
{
    return *_operator;
}

StaticOperator& Operator::getStaticOperator()
{
    return *_operator;
}

bool operator==(const Operator& o, const StaticOperator& so)
{
    return o.getStaticOperator() == so;
}

bool operator==(const StaticOperator& so, const Operator& o)
{
    return o.getStaticOperator() == so;
}

}

}


