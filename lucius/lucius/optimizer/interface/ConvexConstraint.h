/*! \file   ConvexConstraint.h
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Sunday March 9, 2014
	\brief  The header file for the ConvexConstrain class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimizer/interface/Constraint.h>

namespace lucius
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
	virtual void apply(Matrix& ) const;

public:
	virtual Constraint* clone() const;

};

}

}
