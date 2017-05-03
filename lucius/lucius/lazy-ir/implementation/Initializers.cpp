/*  \file   Initializers.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the initializer interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/Initializers.h>

namespace lucius
{

namespace lazy
{

LazyValue createInitializer(std::function<LazyValue()> body)
{
    auto& builder = getBuilder();

    builder.saveInsertionPoint();

    builder.newInitializationFunction();

    auto value = body();

    auto variable = builder.registerValueAsVariable(value);

    builder.restoreInsertionPoint();

    return LazyValue(variable);
}

}

}

