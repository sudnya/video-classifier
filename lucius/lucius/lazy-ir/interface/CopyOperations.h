/*  \file   CopyOperations.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the lazy copy operation interface functions.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace lazy { class LazyValue; } }

namespace lucius
{

namespace lazy
{

/*! \brief Copy from one value into another. */
LazyValue copy(LazyValue input);

/*! \brief Copy from one value into another. */
void copy(LazyValue& output, LazyValue input);

}

}



