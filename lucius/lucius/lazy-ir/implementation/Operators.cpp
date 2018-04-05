/*  \file   Operators.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Operators class family.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/Operators.h>

#include <lucius/ir/values/interface/ConstantOperator.h>

#include <lucius/matrix/interface/GenericOperators.h>

namespace lucius
{

namespace lazy
{

Operator::Operator(const matrix::Operator& op)
: _operator(op)
{

}

const matrix::Operator& Operator::getOperator() const
{
    return _operator;
}

BinaryOperator::BinaryOperator(const matrix::Operator& op)
: Operator(op)
{

}

UnaryOperator::UnaryOperator(const matrix::Operator& op)
: Operator(op)
{

}

Add::Add()
: BinaryOperator(matrix::Add())
{

}

Multiply::Multiply()
: BinaryOperator(matrix::Multiply())
{

}

}

}





