/*! \file   LinearConstraint.cpp
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Sunday March 9, 2014
	\brief  The source file for the LinearConstrain class.
*/

// Lucius Includes
#include <lucius/optimizer/interface/LinearConstraint.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace optimizer
{

LinearConstraint::~LinearConstraint()
{
	
}

bool LinearConstraint::isSatisfied(const Matrix& ) const
{
	assertM(false, "Not implemented.");
}

void LinearConstraint::apply(Matrix& ) const
{
	assertM(false, "Not implemented.");
}

Constraint* LinearConstraint::clone() const
{
	return new LinearConstraint(*this);
}

}

}

