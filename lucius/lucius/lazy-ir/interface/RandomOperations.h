/*  \file   RandomOperations.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the lazy random operation interface functions.
*/

#pragma once

// Standard Library Includes
#include <cstddef>

// Forward Declarations
namespace lucius { namespace lazy { class LazyValue; } }

namespace lucius { namespace matrix { class Dimension; } }
namespace lucius { namespace matrix { class Precision; } }

namespace lucius
{

namespace lazy
{

// Namespace imports
using Dimension = matrix::Dimension;
using Precision = matrix::Precision;

LazyValue srand(size_t seed);

LazyValue rand( LazyValue state, const Dimension& shape, const Precision& precision);
LazyValue randn(LazyValue state, const Dimension& shape, const Precision& precision);

}

}




