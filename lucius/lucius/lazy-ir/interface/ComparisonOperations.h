/*  \file   ComparisonOperations.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the lazy comparison operation interface functions.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace lazy { class LazyValue; } }

namespace lucius
{

namespace lazy
{

/*! \brief Return true if the left is less than the right. */
LazyValue lessThan(LazyValue left, LazyValue right);

}

}




