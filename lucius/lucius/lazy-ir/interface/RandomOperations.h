/*  \file   RandomOperations.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the lazy random operation interface functions.
*/

#pragma once

// Forward Declarations
class lucius { class lazy { class LazyValue; } }

namespace lucius
{

namespace lazy
{

LazyValue srand(size_t seed);

LazyValue rand( LazyValue& state, const Dimension& shape, const Precision& precision);
LazyValue randn(LazyValue& state, const Dimension& shape, const Precision& precision);

}

}




