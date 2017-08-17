/*  \file   Initializers.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the initializer interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/Initializers.h>

#include <lucius/lazy-ir/interface/LazyIr.h>
#include <lucius/lazy-ir/interface/LazyValue.h>

#include <lucius/ir/interface/IRBuilder.h>
#include <lucius/ir/interface/Variable.h>

namespace lucius
{

namespace lazy
{

LazyValue createInitializer(std::function<LazyValue()> body)
{
    auto& builder = getBuilder();

    builder.saveInsertionPoint();

    builder.addInitializationFunction();

    auto value = body();

    auto variable = builder.registerValueAsVariable(value.getValue());

    builder.restoreInsertionPoint();

    return LazyValue(variable);
}

}

}

