/*  \file   RandomOperations.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the lazy random operation interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/RandomOperations.h>

#include <lucius/lazy-ir/interface/LazyValue.h>
#include <lucius/lazy-ir/interface/LazyIr.h>

#include <lucius/ir/interface/IRBuilder.h>
#include <lucius/ir/interface/Constant.h>

namespace lucius
{

namespace lazy
{

LazyValue srand(size_t seed)
{
    auto& builder = getBuilder();

    return LazyValue(builder.addSrand(builder.addConstant(seed)));
}

LazyValue rand(LazyValue& state, const Dimension& shape, const Precision& precision)
{
    auto& builder = getBuilder();

    auto result = builder.addRand(state.getValue(), builder.getTensorType(shape, precision));

    state = LazyValue(builder.addGet(result, builder.addConstant(1)));

    return LazyValue(builder.addGet(result, builder.addConstant(0)));
}

LazyValue randn(LazyValue& state, const Dimension& shape, const Precision& precision)
{
    auto& builder = getBuilder();

    auto result = builder.addRand(state.getValue(), builder.getTensorType(shape, precision));

    state = LazyValue(builder.addGet(result, builder.addConstant(1)));

    return LazyValue(builder.addGet(result, builder.addConstant(0)));
}

}

}





