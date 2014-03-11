/*! \file   ConstantConstraint.h
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Sunday March 9, 2014
	\brief  The header file for the ConstantConstraint class.
*/

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/LinearConstraint.h>

namespace minerva
{

namespace optimizer
{

/*! \brief A constraint for an optimization problem */
class ConstantConstraint : public LinearConstraint
{
public:
	ConstantConstraint(float value);

public:
	virtual ~ConstantConstraint();

public:
	/*! \brief Does the solution satisfy the specified constraint */
	virtual bool isSatisfied(const Matrix& ) const; 

public:
	virtual void apply(BlockSparseMatrix& ) const;

public:
	virtual Constraint* clone() const;

private:
	float _value;

};

}

}


