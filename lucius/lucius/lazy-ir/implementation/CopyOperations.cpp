/*  \file   CopyOperations.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the lazy copy operation interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/CopyOperations.h>

namespace lucius
{

namespace lazy
{

LazyValue copy(LazyValue input)
{
    auto& buidler = getBuilder();

    LazyValue returnValue(builder.newValue(input.getType()));

    copy(returnValue, input);

    return returnValue;
}

void copy(LazyValue output, LazyValue input)
{
    auto& buidler = getBuilder();

    builder.addCopy(output.getValue(), input.getValue());
}

}

}




