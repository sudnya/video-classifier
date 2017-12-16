/*  \file   Operators.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Operators class family.
*/

#pragma once

// Standard Library Includes
#include <cstddef>

namespace lucius
{

namespace lazy
{

class Operator
{
public:
    Operator(size_t id);

public:
    size_t getId() const;

private:
    size_t _id;

};

class BinaryOperator : public Operator
{
public:
    BinaryOperator(size_t id);

};

class UnaryOperator : public Operator
{
public:
    UnaryOperator(size_t id);

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




