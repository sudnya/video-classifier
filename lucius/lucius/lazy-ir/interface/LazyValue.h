/*  \file   LazyValue.h
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The header file for the LazyValue class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class Value; } }

namespace lucius
{

namespace lazy
{

/*! \brief Represents the result of a lazy computation. */
class LazyValue
{
public:
    explicit LazyValue(ir::Value* );

public:
    template <typename T>
    T materialize();

public:
    ir::Value* getValue();

private:
    void* _runProgram();

private:
    ir::Vaue* _value;
};

}

}

#include <lucius/lazy-ir/interface/LazyValue.inl>

