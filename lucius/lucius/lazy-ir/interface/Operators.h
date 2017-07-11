/*  \file   Operators.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Operators class family.
*/

#pragma once

namespace lucius
{

namespace lazy
{

class Operator
{
public:
    size_t getId() const;

};

class BinaryOperator : public Operator
{

};

class UnaryOperator : public Operator
{

};

class Add : public BinaryOperator
{

};

}

}




