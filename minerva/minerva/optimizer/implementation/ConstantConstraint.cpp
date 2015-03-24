/*! \file   ConstantConstraint.cpp
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Sunday March 9, 2014
	\brief  The source file for the ConstantConstraint class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/ConstantConstraint.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace optimizer
{

ConstantConstraint::ConstantConstraint(float value, Comparison c)
: _value(value), _comparison(c)
{

}

ConstantConstraint::~ConstantConstraint()
{

}

bool ConstantConstraint::isSatisfied(const Matrix& m) const
{
	if(_comparison == LessThanOrEqual)
	{
		return reduce(apply(m, LessThanOrEqual(_value)), {}, Sum()) == 0.0;
	}
	else if(_comparison == GreaterThanOrEqual)
	{
		return reduce(apply(m, GreaterThanOrEqual(_value)), {}, Sum()) == 0.0;
	}
	else
	{
		assertM(false, "not implemented");
	}
	
	return false;
} 

void ConstantConstraint::apply(Matrix& m) const
{
	if(_comparison == LessThanOrEqual)
	{
		apply(m, m, Min(_value));
	}
	else if(_comparison == GreaterThanOrEqual)
	{
		apply(m, m, Max(_value));
	}
	else
	{
		assertM(false, "not implemented");
	}
}

Constraint* ConstantConstraint::clone() const
{
	return new ConstantConstraint(*this);
}

}

}
