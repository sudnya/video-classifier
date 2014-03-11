/*! \file   ConvexConstraint.h
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Sunday March 9, 2014
	\brief  The header file for the ConvexConstrain class.
*/

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/Constraint.h>

namespace minerva
{

namespace optimizer
{

/*! \brief A constraint for an optimization problem */
class ConvexConstraint : public Constraint
{
public:
	typedef matrix::Matrix Matrix;

public:
	virtual ~ConvexConstraint();

public:
	/*! \brief Does the solution satisfy the specified constraint */
	virtual bool isSatisfied(const Matrix& ) const; 

public:
	virtual void apply(BlockSparseMatrix& ) const;

public:
	virtual Constraint* clone() const;

};

}

}
