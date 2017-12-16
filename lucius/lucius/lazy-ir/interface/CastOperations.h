/*  \file   CastOperations.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the lazy cast operation interface functions.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace lazy { class LazyValue; } }

namespace lucius
{

namespace lazy
{

/*! \brief Cast to a scalar value. */
LazyValue castToScalar(LazyValue input);

}

}




