/*  \file   ComparisonOperations.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the lazy comparison operation interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/LazyIr.h>

#include <lucius/lazy-ir/interface/LazyValue.h>

#include <lucius/ir/interface/IRBuilder.h>
#include <lucius/lazy-ir/interface/ComparisonOperations.h>

namespace lucius
{

namespace lazy
{

/*! \brief Return true if the left is less than the right. */
LazyValue lessThan(LazyValue left, LazyValue right)
{
    auto& builder = getBuilder();

    return LazyValue(builder.addLessThan(left.getValueForRead(), right.getValueForRead()));
}

}

}





