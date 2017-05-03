/*  \file   Initializers.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the initializer interface functions.
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

LazyValue createInitializer(std::function<LazyValue()> body);

}

}


