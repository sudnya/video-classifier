/*  \file   LazyValue.h
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The header file for the LazyValue class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Value.h>

namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{

namespace lazy
{

/*! \brief Represents the result of a lazy computation. */
class LazyValue
{
public:
    LazyValue();
    explicit LazyValue(ir::Value );

public:
    template <typename T>
    T materialize();

    matrix::Matrix materialize();

public:
          ir::Value& getValue();
    const ir::Value& getValue() const;

private:
    void* _runProgram();

private:
    ir::Value _value;
};

}

}

#include <lucius/lazy-ir/implementation/LazyValue.inl>

