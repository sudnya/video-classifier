/*  \file   Loops.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the loop interface functions.
*/

#pragma once

// Standard Library Includes
#include <function>

// Forward Declarations
class lucius { class lazy { class LazyValue; } }

namespace lucius
{

namespace lazy
{

void forLoop(std::function<LazyValue()> initializer,
    std::function<LazyValue()> condition,
    std::function<LazyValue()> increment,
    std::function<void()> body);

void forLoop(size_t iterations, std::function<void()> f);

}

}


