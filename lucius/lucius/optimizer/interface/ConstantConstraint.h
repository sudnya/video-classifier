/*! \file   ConstantConstraint.h
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \date   Sunday March 9, 2014
    \brief  The header file for the ConstantConstraint class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimizer/interface/LinearConstraint.h>

namespace lucius
{

namespace optimizer
{

/*! \brief A constraint for an optimization problem */
class ConstantConstraint : public LinearConstraint
{
public:
    enum Comparison
    {
        GreaterThanOrEqual,
        LessThanOrEqual
    };

public:
    ConstantConstraint(float value, Comparison c = LessThanOrEqual);

public:
    virtual ~ConstantConstraint();

public:
    /*! \brief Does the solution satisfy the specified constraint */
    virtual bool isSatisfied(const Matrix& ) const; 

public:
    virtual void apply(Matrix& ) const;

public:
    virtual Constraint* clone() const;

private:
    float      _value;
    Comparison _comparison;

};

}

}


