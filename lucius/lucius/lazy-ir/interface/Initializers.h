/*  \file   Initializers.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the initializer interface functions.
*/

#pragma once

// Standard Library Includes
#include <functional>

// Forward Declarations
namespace lucius { namespace lazy { class LazyValue; } }

namespace lucius
{

namespace lazy
{

LazyValue createInitializer(std::function<LazyValue()> body);

}

}


