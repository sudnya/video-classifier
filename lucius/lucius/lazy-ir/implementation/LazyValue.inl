/*  \file   LazyValue.inl
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The source header file for the LazyValue class.
*/

#pragma once

// Lucius Includes
#include <lucius/lazy-ir/interface/LazyValue.h>

// Standard Library Includes
#include <stdexcept>

namespace lucius
{

namespace lazy
{

template <typename T>
T LazyValue::materialize()
{
    void* value = _runProgram();

    T* result = reinterpret_cast<T*>(value);

    if(value == nullptr)
    {
        throw std::runtime_error("Could not get return value from IR execution engine.");
    }

    T resultValue = *result;

    _clearState();

    return resultValue;
}

}

}

