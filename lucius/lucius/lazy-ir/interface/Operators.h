/*  \file   Operators.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Operators class family.
*/

#pragma once

// Lucius Includes
#include <lucius/matrix/interface/Operator.h>

// Standard Library Includes
#include <cstddef>

namespace lucius
{

namespace lazy
{

class Operator
{
public:
    Operator(const matrix::Operator& id);

public:
    const matrix::Operator& getOperator() const;

private:
    matrix::Operator _operator;

};

class BinaryOperator : public Operator
{
public:
    BinaryOperator(const matrix::Operator& );

};

class UnaryOperator : public Operator
{
public:
    UnaryOperator(const matrix::Operator& );

};

class Add : public BinaryOperator
{
public:
    Add();

};

class Multiply : public BinaryOperator
{
public:
    Multiply();

};

}

}




