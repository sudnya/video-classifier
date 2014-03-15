/*! \file   ConstantConstraint.cpp
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Sunday March 9, 2014
	\brief  The source file for the ConstantConstraint class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/ConstantConstraint.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>

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
		return m.greaterThanOrEqual(_value).reduceSum() == 0;
	}
	else if(_comparison == GreaterThanOrEqual)
	{
		return m.lessThanOrEqual(_value).reduceSum() == 0;
	}
	else
	{
		assertM(false, "not implemented");
	}
	
	return false;
} 

void ConstantConstraint::apply(BlockSparseMatrix& m) const
{
	if(_comparison == LessThanOrEqual)
	{
		m.minSelf(_value);
	}
	else if(_comparison == GreaterThanOrEqual)
	{
		m.maxSelf(_value);
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
