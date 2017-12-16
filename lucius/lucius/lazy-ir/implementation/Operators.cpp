/*  \file   Operators.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Operators class family.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/Operators.h>

#include <lucius/ir/values/interface/ConstantOperator.h>

namespace lucius
{

namespace lazy
{

Operator::Operator(size_t id)
: _id(id)
{

}

size_t Operator::getId() const
{
    return _id;
}

BinaryOperator::BinaryOperator(size_t id)
: Operator(id)
{

}

UnaryOperator::UnaryOperator(size_t id)
: Operator(id)
{

}

Add::Add()
: BinaryOperator(ir::Add().getOperatorId())
{

}

Multiply::Multiply()
: BinaryOperator(ir::Multiply().getOperatorId())
{

}

}

}





