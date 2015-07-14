/*! \file   ConvexConstraint.cpp
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \date   Sunday March 9, 2014
    \brief  The source file for the ConvexConstraint class.
*/

// Lucious Includes
#include <lucious/optimizer/interface/ConvexConstraint.h>

#include <lucious/util/interface/debug.h>

namespace lucious
{

namespace optimizer
{

ConvexConstraint::~ConvexConstraint()
{

}

bool ConvexConstraint::isSatisfied(const Matrix& ) const
{
    assertM(false, "Not implemented.");
}

void ConvexConstraint::apply(Matrix& ) const
{
    assertM(false, "Not implemented.");
}

Constraint* ConvexConstraint::clone() const
{
    return new ConvexConstraint(*this);
}

}

}


