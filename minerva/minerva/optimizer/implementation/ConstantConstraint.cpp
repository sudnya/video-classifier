/*! \file   ConstantConstraint.cpp
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Sunday March 9, 2014
	\brief  The source file for the ConstantConstraint class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/ConstantConstraint.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>

namespace minerva
{

namespace optimizer
{

ConstantConstraint::ConstantConstraint(float value)
: _value(value)
{

}

ConstantConstraint::~ConstantConstraint()
{

}

bool ConstantConstraint::isSatisfied(const Matrix& m) const
{
	return m.lessThanOrEqual(_value).reduceSum() == 0;
} 

void ConstantConstraint::apply(BlockSparseMatrix& m) const
{
	m.minSelf(_value);
}

Constraint* ConstantConstraint::clone() const
{
	return new ConstantConstraint(*this);
}

}

}
