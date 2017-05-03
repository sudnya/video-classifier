/*  \file   RandomOperations.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the lazy random operation interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/RandomOperations.h>

namespace lucius
{

namespace lazy
{

LazyValue srand(size_t seed)
{
    auto& buidler = getBuilder();

    LazyValue returnValue(builder.getRandomStateType());

    builder.addSrand(returnValue, builder.getConstant(seed));

    return returnValue;
}

LazyValue rand( LazyValue& state, const Dimension& shape, const Precision& precision)
{
    auto& buidler = getBuilder();

    LazyValue returnValue(builder.newValue(builder.getTensorType(shape, precision)));

    builder.addRand(returnValue.getValue(), state.getValue());

    return returnValue;

}

LazyValue randn(LazyValue& state, const Dimension& shape, const Precision& precision)
{
    auto& buidler = getBuilder();

    LazyValue returnValue(builder.newValue(builder.getTensorType(shape, precision)));

    build.addRandn(returnValue.getValue(), state.getValue());

    return returnValue;
}

}

}





