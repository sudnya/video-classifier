/*! \file   ConvexConstraint.cpp
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Sunday March 9, 2014
	\brief  The source file for the ConvexConstraint class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/ConvexConstraint.h>

#include <minerva/util/interface/debug.h>

namespace minerva
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

void ConvexConstraint::apply(BlockSparseMatrix& ) const
{
	assertM(false, "Not implemented.");
}

Constraint* ConvexConstraint::clone() const
{
	return new ConvexConstraint(*this);
}

}

}


