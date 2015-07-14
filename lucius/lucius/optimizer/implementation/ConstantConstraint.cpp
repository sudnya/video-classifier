/*! \file   ConstantConstraint.cpp
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \date   Sunday March 9, 2014
    \brief  The source file for the ConstantConstraint class.
*/

// Lucious Includes
#include <lucious/optimizer/interface/ConstantConstraint.h>

#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixOperations.h>
#include <lucious/matrix/interface/Operation.h>

#include <lucious/util/interface/debug.h>

namespace lucious
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
        return reduce(matrix::apply(m, matrix::LessThanOrEqual(_value)), {}, matrix::Add())[0] == 0.0;
    }
    else if(_comparison == GreaterThanOrEqual)
    {
        return reduce(matrix::apply(m, matrix::GreaterThanOrEqual(_value)), {}, matrix::Add())[0] == 0.0;
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
        matrix::apply(m, m, matrix::Minimum(_value));
    }
    else if(_comparison == GreaterThanOrEqual)
    {
        matrix::apply(m, m, matrix::Maximum(_value));
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
