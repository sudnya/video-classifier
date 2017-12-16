/*  \file   CastOperations.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the lazy cast operation interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/CastOperations.h>

#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Type.h>

#include <lucius/lazy-ir/interface/LazyValue.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace lazy
{

/*! \brief Cast to a scalar value. */
LazyValue castToScalar(LazyValue input)
{
    if(input.getValue().getType().isScalar())
    {
        return input;
    }

    assertM(false, "Not implemented.");
}

}

}





