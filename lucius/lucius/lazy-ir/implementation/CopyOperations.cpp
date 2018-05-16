/*  \file   CopyOperations.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the lazy copy operation interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/CopyOperations.h>

#include <lucius/lazy-ir/interface/LazyIr.h>
#include <lucius/lazy-ir/interface/LazyValue.h>

#include <lucius/ir/interface/IRBuilder.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace lazy
{

LazyValue copy(LazyValue input)
{
    auto& builder = getBuilder();

    return LazyValue(builder.addCopy(input.getValueForRead()));
}

void copy(LazyValue output, LazyValue input)
{
    auto& builder = getBuilder();

    output.addDefinition(builder.addCopy(input.getValueForRead()));
}

}

}




